import os
import json
from openai import OpenAI, RateLimitError
from tqdm import tqdm
from threading import Thread
from queue import Queue
import time



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


def spin_thread(segments, results):
    """
    Generate sentiment descriptions for each segment in parallel.

    Args:
        segments (list): A list of segments, where each segment is a dictionary containing a 'line' key.
        results (Queue): A queue to store the output segments.

    Returns:
        None
    """
    gen = DescriptionGenerator()

    def try_gen(line):
        try:
            sentiment = gen.generate(line)
        except RateLimitError:
            print("Rate limit reached. Waiting 10 seconds...")
            time.sleep(1)
            sentiment = try_gen(line)
        return sentiment

    output = segments.copy()
    for segment in output:
        sentiment = try_gen(segment['line'])
        segment['sentiment'] = sentiment

    results.put(output)


def threaded_create_segment_descriptions(path, num_threads=16, debug=False):
    """
    Create segment descriptions using multiple threads.

    Args:
        path (str): The path to the directory containing the segments.json file.
        num_threads (int, optional): The number of threads to use for processing. Defaults to 16.
        debug (bool, optional): Whether to run in debug mode. If True, only a subset of segments will be processed. Defaults to False.
    """
    full_path = path + '/segments.json'

    with open(full_path, 'r') as f:
        segments = json.load(f)
        print(f"Num segments: {len(segments)}")
        
        if debug:
            segments = segments[0:200]

        seg_batch = len(segments)//num_threads

        threads = []

        # make thread safe queue
        results = Queue()

        # make threads
        for i in range(0, num_threads):
            
            if i == num_threads - 1:
                segment = segments[i*seg_batch:]
                print("Segment batch: ", i*seg_batch, len(segments))
            else:
                segment = segments[i*seg_batch: (i+1)*seg_batch]
                print("Segment batch: ", i*seg_batch, (i+1)*seg_batch)

            thread = Thread(target=spin_thread,
                            args=(segment, results))
            thread.start()
            threads.append(thread)
            print(f"Thread {i} started")
            
        
        # join threads
        loop = tqdm(threads)
        for thread in loop:
            thread.join()

    # convert the queue to a list
    output = []

    for result in list(results.queue):
        output.extend(result)

    # write to the output file
    full_path = path + '/segments_with_descriptions.json'

    with open(full_path, 'w') as f:
        if debug:
            json.dump(output, f, indent=4)
        else:
            json.dump(output, f)


# depricated
def create_segment_descriptions(path, debug=False):
    gen = DescriptionGenerator()

    full_path = path + '/segments.json'

    with open(full_path, 'r') as f:
        segments = json.load(f)

        if debug:
            segments = segments[0:5]

        loop = tqdm(segments)
        for segment in loop:
            sentiment = gen.generate(segment['line'])
            segment['sentiment'] = sentiment
            loop.update(1)

    full_path = path + '/segments_with_descriptions.json'

    with open(full_path, 'w') as f:
        if debug:
            json.dump(segments, f, indent=4)
        else:
            json.dump(segments, f)

    