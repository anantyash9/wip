
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

class HelloWorld(Resource):
    def get(self):
        start_new_thread(main.main,('Catching_tuna_Maldivian_style.mp4','00:01:25.00','00:01:58.00'))
        
    
@app.route('/progress')
def return_progress():
    return str(main.FishingDetection.progress)
    
api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(debug=True)