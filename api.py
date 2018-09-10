
# coding: utf-8

# In[1]:


import main
from _thread import start_new_thread
from requests.models import Response


# In[2]:





# In[1]:


from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class ActivityProcess(Resource):
    def get(self,video,start,end):
        start_new_thread(main.main,(video,start,end))
        
        
class ActivityProgress(Resource):
    def get(self):
        return (main.FishingDetection.progress)
    
api.add_resource(ActivityProcess, '/activity/process/<video>,<start>,<end>',methods=['GET','POST'])
api.add_resource(ActivityProgress, '/activity/progress/',methods=['GET','POST'])

if __name__ == '__main__':
    app.run(debug=True)
