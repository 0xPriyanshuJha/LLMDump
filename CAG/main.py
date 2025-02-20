from fastapi import FastAPI, Query
import redis
import ollama
import json
import uvicorn

# Initializing Redis Client
redis_client = redis.Redis(host='local', port=6379, db=0, decode_responses=True)

# Initializing App
app = FastAPI()

# Model Configuration
def generarte_response(query:str)->str:
    response = ollama.chat(model="llama3.1", messages=[{"role":"user","content":query}])
    return response["messages"]["content"]


# Response from cached source
@app.get("/response")
def generate(query:str=Query(..., description="User Query")):
    # check if the query is already in cached source
    cached_response = redis_client.get(query)
    if cached_response:
        return {"response": cached_response, "cached":True}
    
    # generating new response
    response = generate(query)

    # store response in cache expiry 1 hr
    redis_client.setex(query, 3600, response)


if __name__ =="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)