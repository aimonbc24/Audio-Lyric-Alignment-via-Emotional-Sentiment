import os
import json
from openai import OpenAI
from tqdm import tqdm


class DescriptionGenerator:
    """
    A class that generates a description of the sentiment of a paragraph of lyrics from a song.
    """

    def __init__(self):
        """
        Initializes the DescriptionGenerator class.

        Parameters:
        None

        Returns:
        None
        """
        key = os.getenv('OPENAI_API_KEY')

        if key is None:
            raise ValueError("API key not found. Please make sure to set the OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=key)


    def generate(self, lyrics):
        """
        Generates a description of the sentiment of a paragraph of lyrics from a song.

        Parameters:
        - lyrics (str): The paragraph of lyrics from a song.

        Returns:
        - response (str): The generated description of the sentiment of the lyrics.
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "user", "content": f"Here's a paragraph of lyrics from a song: \"{lyrics}\" Generate a 2 sentence description of the sentiment of this paragraph? DO NOT make it conversational. Be turse. Start it with: The lyrics are..."},
            ]
        )
        response = response.choices[0].message.content
        return response


def create_segment_descriptions(debug=False):
    gen = DescriptionGenerator()

    with open('../data/segments.json', 'r') as f:
        segments = json.load(f)

        if debug:
            segments = segments[0:5]

        loop = tqdm(segments)
        for segment in loop:
            sentiment = gen.generate(segment['line'])
            segment['sentiment'] = sentiment
            loop.update(1)


    with open('../data/segments_with_descriptions.json', 'w') as f:
        if debug:
            json.dump(segments, f, indent=4)
        else:
            json.dump(segments, f)

    