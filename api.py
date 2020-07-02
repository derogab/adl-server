from constants import Constants
import urllib.request, json 

class API:

    def __init__(self):
        # Init data
        self.activities = None
        self.positions = None
        self.form = None

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
            if activities is not None:
                self.activities = activities

        # Return activities
        return self.activities

    # Function to get device positions informations
    def get_positions(self):

        if self.positions is None:
            # Download device positions
            positions = []
            with urllib.request.urlopen(Constants.api_path + "/positions") as url:
                data = json.loads(url.read().decode())
                # Get device positions
                positions = data['positions']
            # Save device positions
            if positions is not None:
                self.positions = positions

        # Return device positions
        return self.positions

    # Function to get form informations
    def get_form(self):

        if self.form is None:
            # Download form
            form = []
            with urllib.request.urlopen(Constants.api_path + "/form") as url:
                data = json.loads(url.read().decode())
                # Get form
                form = data['groups']
            # Save form
            if form is not None:
                self.form = form

        # Return form
        return self.form