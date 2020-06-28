from constants import Constants
import urllib.request, json 

class API:

    def __init__(self):
        # Init data
        self.activities = None

    # Function to get activity informations
    def get_activities(self):

        if self.activities is None:
            # Download activities
            activities = []
            with urllib.request.urlopen(Constants.api_path + "/activities") as url:
                data = json.loads(url.read().decode())
                # Get activities
                activities = data['activities']
            # Save activities
            self.activities = activities

        # Return activities
        return self.activities