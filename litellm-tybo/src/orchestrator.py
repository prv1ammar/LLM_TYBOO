"""
orchestrator.py — General-Purpose Automation Agent
====================================================
PURPOSE:
  A fully unrestricted AI agent with real tools attached.
  Unlike a plain LLM, this agent can actually DO things:
    - Call external APIs and webhooks
    - Execute Python code for data processing
    - Read and write files
    - Send Slack messages and emails
    - Search the internal knowledge base (RAG)

NO DOMAIN RESTRICTIONS:
  The agent is intentionally unrestricted — it handles any task:
  ETL pipelines, document analysis, code generation, API orchestration,
  contract review, data transformation, automation workflows, etc.

HOW TOOLS WORK WITH PYDANTICAI:
  Tools are Python functions decorated with @agent.tool_plain.
  When the LLM decides it needs to call a tool, PydanticAI:
    1. Extracts the function arguments from the LLM output
    2. Calls the actual Python function with those arguments
    3. Feeds the return value back to the LLM as context
    4. The LLM continues generating its final response

  This loop can happen multiple times in a single agent.run() call.

AVAILABLE TOOLS:
  search_knowledge_base  — RAG search over ingested documents
  http_get               — HTTP GET to any URL
  http_post              — HTTP POST for API calls and webhooks
  trigger_webhook        — Send structured event to a webhook URL
  send_email             — Queue an email (plug SMTP in .env)
  post_to_slack          — Post to Slack (requires SLACK_WEBHOOK_URL)
  run_python             — Execute a Python snippet in a subprocess
  run_shell              — Execute a shell command
  read_file              — Read a local file into context
  write_file             — Write content to a local file
  parse_json             — Parse and format a JSON string

SPECIALIZED METHODS:
  analyze_document()  — Deep analysis with structured output (AnalysisResult)
  generate_code()     — Code generation for any language
  review_code()       — Code review with quality score (CodeReviewResult)
  run_etl()           — ETL pipeline execution (ETLResult)

HOW TO USE:
  from orchestrator import AgentOrchestrator

  orchestrator = AgentOrchestrator()

  # General query with auto tool use
  result = await orchestrator.run_agent(
      "Fetch the latest EUR/MAD rate from https://api.exchangerate.host/latest?base=EUR&symbols=MAD"
  )

  # Structured document analysis
  analysis = await orchestrator.analyze_document(contract_text)
  print(analysis.summary)
  print(analysis.key_points)

  # Code generation
  code = await orchestrator.generate_code(
      task="Parse a CSV file, filter rows where amount > 1000, export as JSON",
      language="python"
  )

  # ETL pipeline
  result = await orchestrator.run_etl(
      source_data=csv_string,
      transformation_rules="Keep only paid transactions, sum totals by user",
      output_format="json"
  )
"""

import os
import json
import httpx
import subprocess
import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from pydantic_ai import Agent
from model_router import get_model_for_chat, get_model_for_rag
from dotenv import load_dotenv

load_dotenv()


# ── Structured output models ──────────────────────────────────────────────────
# These Pydantic models define the exact shape of the data returned
# by specialized agent methods. PydanticAI enforces the structure.

class AnalysisResult(BaseModel):
    """
    Returned by analyze_document().
    Fields:
      summary        → 2-4 sentence overview of the document
      key_points     → list of important facts or clauses
      sentiment      → "positive", "negative", "neutral", or "mixed"
      confidence     → float from 0.0 to 1.0 (how confident the model is)
      extracted_data → any key-value data extracted (dates, amounts, names, etc.)
    """
    summary: str
    key_points: List[str]
    sentiment: str
    confidence: float
    extracted_data: Dict[str, Any] = {}


class CodeReviewResult(BaseModel):
    """
    Returned by review_code().
    Fields:
      quality_score → integer 0-100 (overall code quality)
      issues        → list of problems found (bugs, security, style)
      suggestions   → list of improvement recommendations
      approved      → True if code is production-ready, False otherwise
    """
    quality_score: int
    issues: List[str]
    suggestions: List[str]
    approved: bool


class ETLResult(BaseModel):
    """
    Returned by run_etl().
    Fields:
      records_processed → number of records successfully transformed
      records_failed    → number of records that caused errors
      output_summary    → the transformed data or a summary of it
      errors            → list of error messages for failed records
    """
    records_processed: int
    records_failed: int
    output_summary: str
    errors: List[str]


# ── Main orchestrator class ───────────────────────────────────────────────────

class AgentOrchestrator:
    """
    Unrestricted automation agent with a full tool suite.
    Auto-routes queries to 3B or 14B based on complexity.
    All specialized methods always use 14B for quality.
    """

    def __init__(self):
        # Lazy import to avoid circular dependency:
        # production_rag imports model_router, orchestrator also imports model_router
        # Importing ProductionRAG here instead of at the top avoids the issue
        from production_rag import ProductionRAG
        self.rag = ProductionRAG(collection_name="knowledge_base")

    async def run_agent(self, query: str, system_prompt: str = None) -> str:
        """
        Run the general-purpose agent on any query.

        The agent will:
          1. Analyze what the query is asking for
          2. Decide which tools (if any) to call
          3. Call those tools and incorporate the results
          4. Return a final natural language response

        Args:
            query:         The user's request in any language.
            system_prompt: Override the default system prompt if needed.
                           Useful for giving the agent a specific persona
                           or restricting it to a particular task.

        Returns:
            The agent's final text response as a string.

        Example:
            result = await orchestrator.run_agent(
                "Call https://api.example.com/data, parse the JSON, "
                "and save the 'users' array to /app/output/users.json"
            )
        """
        # Auto-select 3B or 14B based on complexity
        model, model_name = get_model_for_chat(query)

        agent = Agent(
            model,
            system_prompt=system_prompt or (
                "You are a powerful general-purpose AI automation assistant. "
                "You have no domain restrictions — handle any task including "
                "data processing, code generation, document analysis, API calls, "
                "ETL pipelines, and automation workflows. "
                "Use the available tools when they would help complete the task. "
                "Always respond in the same language the user used."
            )
        )

        # ── Tool definitions ──────────────────────────────────────────────────
        # Each tool is a regular async Python function.
        # The LLM reads the docstring to understand what the tool does
        # and when to call it. Keep docstrings clear and specific.

        @agent.tool_plain
        async def search_knowledge_base(query: str) -> str:
            """
            Search the internal document knowledge base using semantic similarity.
            Use this when the user asks about internal company documents,
            policies, contracts, manuals, or any ingested content.
            Returns the most relevant document chunks found.
            """
            result = await self.rag.query(query, top_k=3)
            if result.get("used_knowledge_base"):
                return f"Found in knowledge base:\n{result['answer']}"
            return "No relevant documents found in knowledge base."

        @agent.tool_plain
        async def http_get(url: str, headers: Dict[str, str] = None) -> str:
            """
            Make an HTTP GET request to any URL or API endpoint.
            Use this to fetch data from external APIs, webhooks, or web pages.
            Returns the HTTP status code and response body (truncated to 2000 chars).
            """
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.get(url, headers=headers or {})
                    return f"Status: {resp.status_code}\nBody:\n{resp.text[:2000]}"
            except Exception as e:
                return f"HTTP GET failed: {str(e)}"

        @agent.tool_plain
        async def http_post(url: str, payload: Dict[str, Any], headers: Dict[str, str] = None) -> str:
            """
            Make an HTTP POST request to any URL.
            Use this to send data to APIs, trigger external services,
            or post to webhook endpoints.
            Returns the HTTP status code and response body.
            """
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(url, json=payload, headers=headers or {})
                    return f"Status: {resp.status_code}\nBody:\n{resp.text[:2000]}"
            except Exception as e:
                return f"HTTP POST failed: {str(e)}"

        @agent.tool_plain
        async def trigger_webhook(webhook_url: str, event: str, data: Dict[str, Any]) -> str:
            """
            Send a structured event notification to a webhook URL.
            The payload will be: {"event": event_name, "data": {...}, "timestamp": "..."}
            Use this to notify external systems (n8n, Zapier, custom services)
            about events that happened during processing.
            """
            payload = {
                "event": event,
                "data": data,
                "timestamp": datetime.datetime.utcnow().isoformat()
            }
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.post(webhook_url, json=payload)
                    return f"Webhook triggered. Status: {resp.status_code}"
            except Exception as e:
                return f"Webhook failed: {str(e)}"

        @agent.tool_plain
        async def send_email(recipient: str, subject: str, body: str) -> str:
            """
            Send an email to the specified recipient.
            To enable real email sending, set SMTP_HOST, SMTP_USER, SMTP_PASSWORD in .env
            and implement the SMTP logic below the print statement.
            Currently logs to console (simulation mode).
            """
            # To enable real sending, replace this with smtplib or httpx call to SendGrid
            print(f"[EMAIL] To: {recipient} | Subject: {subject}")
            print(f"[EMAIL] Body preview: {body[:100]}...")
            return f"Email queued for {recipient} — Subject: '{subject}'"

        @agent.tool_plain
        async def post_to_slack(channel: str, message: str) -> str:
            """
            Post a message to a Slack channel.
            Requires SLACK_WEBHOOK_URL to be set in your .env file.
            If SLACK_WEBHOOK_URL is not set, the message is logged to console instead.
            channel should be the channel name without '#' (e.g. "general", "alerts").
            """
            webhook = os.getenv("SLACK_WEBHOOK_URL")
            if not webhook:
                # Simulation mode — no webhook configured
                print(f"[SLACK SIMULATION] #{channel}: {message}")
                return f"[Simulated] Posted to #{channel}"
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.post(webhook, json={
                        "channel": f"#{channel}",
                        "text": message
                    })
                    return f"Slack message sent to #{channel}. Status: {resp.status_code}"
            except Exception as e:
                return f"Slack post failed: {str(e)}"

        @agent.tool_plain
        async def run_python(code: str) -> str:
            """
            Execute a Python code snippet in an isolated subprocess.
            Use this for data transformations, calculations, CSV/JSON processing,
            or any task that benefits from running actual Python code.
            The subprocess has a 30-second timeout.
            Returns stdout and stderr output.
            WARNING: Code runs with the current user's permissions.
            """
            try:
                result = subprocess.run(
                    ["python3", "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                output = result.stdout[:3000] if result.stdout else "(no output)"
                errors = f"\nErrors:\n{result.stderr[:500]}" if result.stderr else ""
                return f"Output:\n{output}{errors}"
            except subprocess.TimeoutExpired:
                return "Execution timed out after 30 seconds."
            except Exception as e:
                return f"Execution failed: {str(e)}"

        @agent.tool_plain
        async def run_shell(command: str) -> str:
            """
            Execute a shell command on the server.
            Use for system automation: listing files, checking disk space,
            running scripts, managing processes, etc.
            Returns the exit code and stdout (truncated to 2000 chars).
            WARNING: Runs with the current user's permissions. Use carefully.
            """
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                return f"Exit code: {result.returncode}\nOutput:\n{result.stdout[:2000]}"
            except subprocess.TimeoutExpired:
                return "Command timed out after 30 seconds."
            except Exception as e:
                return f"Shell command failed: {str(e)}"

        @agent.tool_plain
        async def read_file(file_path: str) -> str:
            """
            Read a local file and return its contents as a string.
            Supports any text-based format: .txt, .json, .csv, .py, .md, .yaml, etc.
            Content is truncated to 8000 characters to stay within context limits.
            Use this to load data files, config files, or code files for processing.
            """
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read(8000)
                return f"File: {file_path}\n---\n{content}"
            except FileNotFoundError:
                return f"File not found: {file_path}"
            except Exception as e:
                return f"Failed to read file: {str(e)}"

        @agent.tool_plain
        async def write_file(file_path: str, content: str) -> str:
            """
            Write a string to a local file, creating directories if needed.
            Use this to save generated code, transformed data, reports,
            or any output that should be persisted to disk.
            If the file already exists, it will be overwritten.
            """
            try:
                # Create parent directories if they don't exist
                os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return f"File written successfully: {file_path} ({len(content):,} characters)"
            except Exception as e:
                return f"Failed to write file: {str(e)}"

        @agent.tool_plain
        async def parse_json(raw: str) -> str:
            """
            Parse a raw JSON string and return it pretty-printed.
            Use this to validate and format JSON data received from APIs
            or extracted from documents before further processing.
            Returns an error message if the input is not valid JSON.
            """
            try:
                parsed = json.loads(raw)
                return json.dumps(parsed, indent=2, ensure_ascii=False)
            except json.JSONDecodeError as e:
                return f"Invalid JSON: {str(e)}"

        print(f"[Agent] Using {model_name} | Query: {query[:70]}...")
        result = await agent.run(query)
        return result.output

    async def analyze_document(self, document: str, instructions: str = None) -> AnalysisResult:
        """
        Perform deep structured analysis of a document.

        Always uses 14B model for maximum accuracy.
        Returns a typed AnalysisResult with summary, key points, sentiment,
        confidence score, and any extracted data fields.

        Args:
            document:     The full document text to analyze.
                          Will be truncated to 6000 chars if longer.
            instructions: Optional specific instructions, e.g.:
                          "Focus on payment terms and penalty clauses"
                          "Extract all dates and monetary amounts"

        Returns:
            AnalysisResult with fields: summary, key_points, sentiment,
            confidence (0.0-1.0), extracted_data (dict of key-value pairs)

        Usage:
            analysis = await orchestrator.analyze_document(
                contract_text,
                instructions="Extract all dates, amounts, and party names"
            )
            print(analysis.summary)
            print(analysis.key_points)
            print(analysis.extracted_data)  # {"amount": "150,000 MAD", "deadline": "2025-06-01"}
        """
        model = get_model_for_rag()  # Always 14B for document work
        agent = Agent(
            model,
            result_type=AnalysisResult,
            system_prompt=(
                "You are a precise document analysis engine. "
                "Extract information thoroughly and objectively. "
                "Be specific and structured in your output."
            )
        )
        prompt = (
            f"Analyze this document:\n\n{document[:6000]}\n\n"
            f"{f'Special instructions: {instructions}' if instructions else ''}\n\n"
            "Return a structured analysis with summary, key_points list, "
            "sentiment (positive/negative/neutral/mixed), "
            "confidence score (0.0-1.0), and extracted_data as key-value pairs."
        )
        result = await agent.run(prompt)
        return result.output

    async def generate_code(self, task: str, language: str = "python", context: str = "") -> str:
        """
        Generate production-ready code for any programming task.

        Always uses 14B model for code quality.
        The generated code includes error handling, type hints (where applicable),
        and inline comments explaining what each section does.

        Args:
            task:     Description of what the code should do.
                      Be specific for better results.
            language: Target programming language (default: "python").
                      Works with JavaScript, TypeScript, SQL, Bash, etc.
            context:  Optional constraints or context, e.g.:
                      "Use pandas and avoid external HTTP calls"
                      "Must work with Python 3.9+"

        Returns:
            String containing the complete, runnable code.

        Usage:
            code = await orchestrator.generate_code(
                task="Read a CSV file, group rows by 'category', sum the 'amount' column per group, save result as JSON",
                language="python",
                context="Use pandas. Input file path from sys.argv[1]"
            )
            print(code)
        """
        model = get_model_for_rag()
        agent = Agent(
            model,
            system_prompt=(
                f"You are an expert {language} developer. "
                "Write clean, production-ready code with proper error handling. "
                "Include comments explaining complex sections. "
                "Do not use placeholder comments — write real, working code."
            )
        )
        prompt = (
            f"Task: {task}\n"
            f"Language: {language}\n"
            f"{f'Context/constraints: {context}' if context else ''}\n\n"
            "Write the complete, runnable code:"
        )
        result = await agent.run(prompt)
        return result.output

    async def review_code(self, code: str, language: str = "python") -> CodeReviewResult:
        """
        Review code quality, security, and correctness.

        Always uses 14B model for thorough analysis.
        Returns a typed CodeReviewResult with issues, suggestions, and a quality score.

        Args:
            code:     The source code to review (truncated to 5000 chars).
            language: The programming language for context-aware review.

        Returns:
            CodeReviewResult with:
              quality_score (0-100) — overall code quality rating
              issues        — list of bugs, security vulnerabilities, or bad practices
              suggestions   — list of concrete improvement recommendations
              approved      — True if the code is production-ready

        Usage:
            review = await orchestrator.review_code(my_python_code)
            if not review.approved:
                print("Issues found:")
                for issue in review.issues:
                    print(f"  - {issue}")
        """
        model = get_model_for_rag()
        agent = Agent(
            model,
            result_type=CodeReviewResult,
            system_prompt=(
                f"You are a senior {language} engineer doing a code review. "
                "Check for bugs, security vulnerabilities, performance issues, "
                "and code style problems. Be specific and actionable."
            )
        )
        result = await agent.run(f"Review this {language} code:\n\n{code[:5000]}")
        return result.output

    async def run_etl(self, source_data: str, transformation_rules: str,
                      output_format: str = "json") -> ETLResult:
        """
        Execute an ETL (Extract, Transform, Load) pipeline described in natural language.

        Always uses 14B model for data accuracy.

        Args:
            source_data:          Raw input data as a string (CSV, JSON, TSV, plain text...).
                                  Will be truncated to 4000 chars if longer.
            transformation_rules: Natural language description of what to do with the data.
                                  Example: "Filter rows where status='paid', sum amount by user"
            output_format:        Desired output format: "json", "csv", "markdown", "sql"

        Returns:
            ETLResult with:
              records_processed → count of successfully transformed records
              records_failed    → count of records that caused errors
              output_summary    → the transformed data or a description of it
              errors            → list of error messages

        Usage:
            result = await orchestrator.run_etl(
                source_data="name,amount,status\nAlice,1500,paid\nBob,800,pending",
                transformation_rules="Keep only paid records, calculate total",
                output_format="json"
            )
            print(result.records_processed)  # 1
            print(result.output_summary)     # {"records": [{"name": "Alice", "amount": 1500}], "total": 1500}
        """
        model = get_model_for_rag()
        agent = Agent(
            model,
            result_type=ETLResult,
            system_prompt=(
                "You are a data pipeline engine. "
                "Process data accurately according to the transformation rules. "
                "Count processed and failed records precisely."
            )
        )
        prompt = (
            f"Execute this ETL pipeline:\n\n"
            f"SOURCE DATA:\n{source_data[:4000]}\n\n"
            f"TRANSFORMATION RULES:\n{transformation_rules}\n\n"
            f"OUTPUT FORMAT: {output_format}\n\n"
            "Apply the transformations and return the result."
        )
        result = await agent.run(prompt)
        return result.output

    async def generate_content(self, prompt: str, context: str = "") -> str:
        """
        Generate any type of content (text, reports, summaries, etc.).
        Kept for backward compatibility with the API endpoints in api.py.

        Args:
            prompt:  What to generate.
            context: Optional background information.

        Returns:
            Generated content as a string.
        """
        full_prompt = f"{prompt}\n\nContext: {context}" if context else prompt
        return await self.run_agent(full_prompt)
