import os
from typing import Any, Dict, List, Tuple, Union
import re
import logging
import requests
import json
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.runnables import Runnable
from langchain.prompts.prompt import PromptTemplate
from langchain.llms.base import BaseLLM  # Adjust import if needed; here assumed as BaseLLM substitute
from utils import ReducedOpenAPISpec, get_matched_endpoint  # Assumed helper functions exist

logger = logging.getLogger(__name__)

# In-Context Learning examples for TMDB and Spotify
icl_examples = {
    "tmdb": """Example 1:
Background: The id of Wong Kar-Wai is 12453
User query: give me the latest movie directed by Wong Kar-Wai.
API calling 1: GET /person/12453/movie_credits to get the latest movie directed by Wong Kar-Wai (id 12453)
API response: The latest movie directed by Wong Kar-Wai is The Grandmaster (id 44865), ...
...
""",
    "spotify": """Example 1:
Background: No background
User query: what is the id of album Kind of Blue.
API calling 1: GET /search to search for the album "Kind of Blue"
API response: Kind of Blue's album_id is 1weenld61qoidwYuZ1GESA
...
"""
}

# API Selector prompt template (common for all scenarios)
API_SELECTOR_PROMPT = """You are a planner that plans a sequence of RESTful API calls to assist with user queries against an API.
Another API caller will execute your plan and return the final answer.
Use the provided endpoints and in-context examples to generate a plan.
If you believe the query is already fulfilled, output the final answer directly.

Here are the available APIs:
{endpoints}

Begin!

Background: {background}
User query: {plan}
API calling 1: {agent_scratchpad}"""

# =============================================================================
#                           APISelector Class
# =============================================================================
class APISelector(Runnable):
    llm: BaseLLM
    api_spec: ReducedOpenAPISpec
    scenario: str
    api_selector_prompt: PromptTemplate
    output_key: str = "result"

    # Constants for multi-step planning
    MAX_ITERATIONS: int = 5

    def __init__(self, llm: Any, scenario: str, api_spec: ReducedOpenAPISpec) -> None:
        # Build a simple endpoints description from the provided API spec.
        # For TMDB and Spotify, we assume api_spec was loaded locally (e.g., from specs/tmdb_oas.json).
        # For NetBox, we will fetch the schema remotely.
        self.scenario = scenario.lower()
        self.api_spec = api_spec
        self.api_url = self.get_base_url(self.scenario)
        if self.scenario == "netbox":
            # For NetBox, fetch the schema from the remote endpoint.
            self.api_docs = self.fetch_api_schema(self.scenario)
        else:
            # For TMDB/Spotify, convert the provided api_spec into a dictionary of endpoints.
            self.api_docs = self.convert_spec_to_docs(api_spec)
        # Prepare the prompt using a simple list of endpoints.
        api_name_desc = [f"{endpoint[0]} {endpoint[1].split('.')[0] if endpoint[1] is not None else ''}" 
                         for endpoint in api_spec.endpoints]
        endpoints_str = "\n".join(api_name_desc)
        api_selector_prompt = PromptTemplate(
            template=API_SELECTOR_PROMPT,
            partial_variables={"endpoints": endpoints_str, "icl_examples": icl_examples.get(scenario, "")},
            input_variables=["plan", "background", "agent_scratchpad"],
        )
        super().__init__()
        self.llm = llm
        self.api_selector_prompt = api_selector_prompt
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        # For NetBox, load ChromaDB; for TMDB/Spotify, vector_db is not used.
        self.vector_db = self.store_api_docs_in_chroma() if self.scenario == "netbox" else None

    @property
    def _chain_type(self) -> str:
        return "RestGPT API Selector"

    @property
    def input_keys(self) -> List[str]:
        return ["plan", "background"]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "API response: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "API calling {}: "
    
    @property
    def _stop(self) -> List[str]:
        return [
            f"\n{self.observation_prefix.rstrip()}",
            f"\n\t{self.observation_prefix.rstrip()}",
        ]

    def get_base_url(self, scenario: str) -> str:
        base_urls = {
            "netbox": "https://demo.netbox.dev/api/",
            "tmdb": "https://api.themoviedb.org/3/",
            "spotify": "https://api.spotify.com/v1/"
        }
        return base_urls.get(scenario, "")

    def fetch_api_schema(self, scenario: str) -> Dict[str, Any]:
        """Fetch the API schema remotely (used only for NetBox)."""
        netbox_schema_url = "https://demo.netbox.dev/api/schema/?format=json"
        response = requests.get(netbox_schema_url)
        if response.status_code == 200:
            schema = response.json()
            return schema.get("paths", {})
        else:
            raise RuntimeError(f"Failed to fetch {scenario} API schema: {response.status_code}")

    def convert_spec_to_docs(self, api_spec: ReducedOpenAPISpec) -> Dict[str, Any]:
        """
        Convert a locally loaded API spec (for TMDB/Spotify) into a dictionary format
        that mimics the structure of remote API docs.
        """
        docs = {}
        for endpoint in api_spec.endpoints:
            # Assuming each endpoint tuple is (name, description, docs)
            name, desc, details = endpoint
            docs[name] = {"get": {"description": desc}}  # Simplified conversion.
        return docs          

    def store_api_docs_in_chroma(self) -> Any:
        """Store NetBox API docs in ChromaDB for vector-based retrieval."""
        if self.scenario != "netbox":
            return None
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_or_create_collection(name="netbox_api")
        for path, details in self.api_docs.items():
            for method, metadata in details.items():
                doc_text = f"{method.upper()} {path} - {metadata.get('description', '')}"
                embedding = self.embedder.encode(doc_text).tolist()
                collection.add(
                    ids=[path],
                    embeddings=[embedding],
                    metadatas=[{"method": method.upper(), "path": path, "description": metadata.get('description', '')}]
                )
        return collection

    def retrieve_api_docs(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant API documentation from ChromaDB (NetBox only)."""
        if not self.vector_db:
            return ""
        query_embedding = self.embedder.encode(query).tolist()
        search_results = self.vector_db.query(query_embeddings=[query_embedding], n_results=top_k)
        retrieved_docs = []
        for result in search_results["metadatas"][0]:
            doc_text = f"{result['method']} {result['path']} - {result['description']}"
            retrieved_docs.append(doc_text)
        return "\n".join(retrieved_docs)

    def _construct_scratchpad(self, history: List[Tuple[str, str, str]], instruction: str) -> str:
        """Combine history of planning steps and execution results into a scratchpad."""
        if not history:
            return ""
        scratchpad = ""
        for i, (plan, api_plan, execution_res) in enumerate(history):
            scratchpad += f"Step {i+1} Plan: {plan}\n"
            scratchpad += f"Step {i+1} API Call: {api_plan}\n"
            scratchpad += f"Step {i+1} Result: {execution_res}\n"
        scratchpad += "Current Instruction: " + instruction + "\n"
        return scratchpad

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """
        Multi-step API planning loop:
         - Maintains a history of planning steps.
         - For NetBox, integrates RAG via ChromaDB.
         - For TMDB/Spotify, uses rule-based matching; falls back to LLM if needed.
        """
        history: List[Tuple[str, str, str]] = inputs.get("history", [])
        iterations = 0
        max_iterations = self.MAX_ITERATIONS
        current_plan = inputs["plan"]
        background = inputs.get("background", "")
        instruction = inputs.get("instruction", "")
        
        while iterations < max_iterations:
            # Build scratchpad from history.
            scratchpad = self._construct_scratchpad(history, instruction)
            # For NetBox, retrieve additional context via RAG.
            if self.scenario == "netbox":
                retrieved_docs = self.retrieve_api_docs(current_plan)
                combined_context = scratchpad + "\nRetrieved Docs:\n" + retrieved_docs
            else:
                combined_context = scratchpad

            # Invoke the LLM with the current plan, background, and combined context.
            api_selector_chain = self.api_selector_prompt | self.llm
            chain_output = api_selector_chain.invoke({
                "plan": current_plan,
                "background": background,
                "agent_scratchpad": combined_context
            })
            new_plan = re.sub(r"API calling \d+: ", "", chain_output.content).strip()
            logger.info(f"Iteration {iterations}: API Selector returned: {new_plan}")

            # Check for termination condition (e.g., final answer).
            if "Final Answer:" in new_plan or "No API call needed" in new_plan:
                history.append((current_plan, new_plan, ""))
                return {"result": new_plan}

            # Optionally, check if the new plan matches an endpoint in the API spec.
            if get_matched_endpoint(self.api_spec, new_plan) is not None:
                history.append((current_plan, new_plan, "Executed"))
                return {"result": new_plan}

            # Simulate an execution result (in a full system, this would call the API).
            execution_result = new_plan  # Placeholder: assume the plan is executed and returns itself.
            history.append((current_plan, new_plan, execution_result))
            current_plan = new_plan
            iterations += 1

        # If max iterations reached, return the last plan.
        return {"result": current_plan}

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        return self._call(inputs)
