import os
import json
import logging
import time
import yaml
import spotipy
from langchain_community.utilities import RequestsWrapper
from langchain_openai import ChatOpenAI
from utils import reduce_openapi_spec, ColorPrint
from model import RestGPT
import requests

logger = logging.getLogger(__name__)


def load_config(config_file='config.yaml'):
    """Load configuration from a YAML file and set environment variables."""
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    os.environ["OPENAI_API_KEY"] = config['openai_api_key']
    os.environ["TMDB_ACCESS_TOKEN"] = config['tmdb_access_token']
    os.environ['SPOTIPY_CLIENT_ID'] = config['spotipy_client_id']
    os.environ['SPOTIPY_CLIENT_SECRET'] = config['spotipy_client_secret']
    os.environ['SPOTIPY_REDIRECT_URI'] = config['spotipy_redirect_uri']
    return config


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(ColorPrint())],
    )
    logger.setLevel(logging.INFO)


def load_api_spec(scenario):
    """Load and process the API specification based on the scenario."""
    if scenario == 'tmdb':
        with open("specs/tmdb_oas.json") as f:
            raw_api_spec = json.load(f)
        api_spec = reduce_openapi_spec(raw_api_spec, only_required=False)
        headers = {'Authorization': f'Bearer {os.environ["TMDB_ACCESS_TOKEN"]}'}
    elif scenario == 'spotify':
        with open("specs/spotify_oas.json") as f:
            raw_api_spec = json.load(f)
        api_spec = reduce_openapi_spec(raw_api_spec, only_required=False, merge_allof=True)
        scopes = list(raw_api_spec['components']['securitySchemes']['oauth_2_0']['flows']['authorizationCode']['scopes'].keys())
        access_token = spotipy.util.prompt_for_user_token(scope=','.join(scopes))
        headers = {'Authorization': f'Bearer {access_token}'}
    elif scenario == 'netbox':
        # Fetch the NetBox API schema from its endpoint.
        netbox_schema_url = "https://demo.netbox.dev/api/schema/?format=json"
        response = requests.get(netbox_schema_url)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch NetBox schema: {response.status_code}")
        raw_api_spec = response.json()
        # Process the schema using reduce_openapi_spec
        api_spec = reduce_openapi_spec(raw_api_spec, only_required=False)
        # Set headers as needed (NetBox may require token auth if enabled)
        headers = {}  # Adjust headers if authentication is needed.
    else:
        raise ValueError(f"❌ Unsupported scenario: {scenario}")

    return api_spec, headers


def get_user_input():
    """Handle user input for scenario selection and query."""
    scenario = input("Please select a scenario (TMDB/Spotify): ").strip().lower()
    query_examples = {
        'tmdb': "Give me the number of movies directed by Sofia Coppola",
        'spotify': "Add Summertime Sadness by Lana Del Rey to my first playlist"
    }
    query_example = query_examples.get(scenario, "What movies did Christopher Nolan direct?")
    print(f"Example instruction: {query_example}")
    query = input("Please input an instruction (Press ENTER to use the example instruction): ").strip()
    if not query:
        query = query_example
    return scenario, query


def main():
    config = load_config()
    setup_logging()
    scenario, query = get_user_input()
    logger.info(f"Query: {query}")
    
    # Load API spec and headers for the selected scenario
    api_spec, headers = load_api_spec(scenario)
    requests_wrapper = RequestsWrapper(headers=headers)
    
    # Initialize the LLM (using ChatOpenAI)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0, max_tokens=700)
    
    # Create the RestGPT agent
    rest_gpt = RestGPT(llm, api_spec=api_spec, scenario=scenario, requests_wrapper=requests_wrapper, simple_parser=False)
    
    start_time = time.time()
    rest_gpt.invoke({"query": query})  # Use .invoke() per new API
    logger.info(f"✅ Execution Time: {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
