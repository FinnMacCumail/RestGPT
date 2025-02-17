import requests
import os
# Replace with your TMDB API key
API_KEY = "1370b21e6746a8aa1aff4b6e5b9fd92e"
BASE_URL = "https://api.themoviedb.org/3"


def get_person_details(person_id):
    """Fetch details of a person from TMDB using their person ID."""
    url = f"{BASE_URL}/person/{person_id}"
    params = {"api_key": API_KEY, "append_to_response": "movie_credits,tv_credits"}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def display_person_details(person_data):
    """Display key details about a person."""
    if not person_data:
        print("No data found.")
        return

    print(f"Name: {person_data.get('name')}")
    print(f"Birthday: {person_data.get('birthday')}")
    print(f"Place of Birth: {person_data.get('place_of_birth')}")
    print(f"Biography: {person_data.get('biography')[:500]}...")  # Limit to 500 chars
    print("\nKnown for Movies:")

    for movie in person_data.get("movie_credits", {}).get("cast", [])[:5]:  # Top 5 movies
        print(f" - {movie.get('title')} ({movie.get('release_date', 'N/A')})")

    print("\nKnown for TV Shows:")
    for show in person_data.get("tv_credits", {}).get("cast", [])[:5]:  # Top 5 TV shows
        print(f" - {show.get('name')} ({show.get('first_air_date', 'N/A')})")


if __name__ == "__main__":
    person_id = input("Enter the TMDB Person ID: ").strip()
    person_data = get_person_details(person_id)
    display_person_details(person_data)
