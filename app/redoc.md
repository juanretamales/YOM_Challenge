## Getting Started

On this page, you will find an introduction to the API usage from a developer perspective, 
and a reference to all the API objects and methods.

## Developer Resources

There are several other key resources related to implementing, integrating and extending the YOM platform:
* **[Swagger](https://127.0.0.1/docs)**: an interactive platform listing of this API which adheres to OpenAPI Specification. , 
* **[Documentation](https://127.0.0.1/redoc)**: the explained book or documentation of this API .

### HTTP and REST

The YOM API is organized around REST. Our API has predictable, resource-oriented URLs, and uses HTTP response codes to indicate API errors. We use built-in HTTP features, like Token authentication and HTTP verbs, which are understood by off-the-shelf HTTP clients.

### cross-origin resource

We support cross-origin resource sharing, allowing you to interact securely with our API from a client-side web application. JSON is returned by API responses, including errors (except when another format is requested, e.g. ZIP).

> When making HTTP requests, be sure to use proper header 'Content-Type:application/json'. 

### Use a endpoint

We will make a new call to the api with this url and we will send the apikey in headers as seen in the example.

<details>
    <summary>Show Code</summary>

        import requests
        url = "https://127.0.0.1/predict/?apikey={apikey}"

        payload={
            "danceability": 0,
            "energy": 0,
            "speechiness": 0,
            "acousticness": 0,
            "valence": 0,
            "tempo": 0
        }
        headers = {
            'Accept': 'application/json',
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        print(response.text)
</details>

## Import in Postman

You can directly use the link to import the API into [Postman](https://www.postman.com/):

1. In Postman press Import button
2. In the window press Link button
3. In the field, copy this url ```https://127.0.0.1/openapi.json``` and press Continue button
4. Next press Import button and wait for the message ** Import complete **
