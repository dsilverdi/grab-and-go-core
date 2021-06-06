# Grab and Go Core
Backend Service for Retail Object Detection based on Grab and Go Datesets using Flask

# Installation
set up your python environment and use package manager [pip](https://pip.pypa.io/en/stable/) to install requirement depedency

```bash
pip install -r requirements.txt
```

then install tensorflow object detection [api](https://github.com/tensorflow/models/tree/master/research/object_detection) using our setupapi
```bash
chmod +x setupapi.sh
./setupapi.sh
```

# API Endpoints
## Predict Object
returns json data about prediction acording to base64 image input
* **URL**

  /detection

* **Method:**

  `POST`

* **Request Body**

  raw plain text
  ```
  data:image/jpeg;base64,/9j/{{base64 encoded image}}
  ```

* **Success Response:**

  * **Code:** 200 <br />
    **Content:** 
    ```json
    [
        {
            "id": 67784,
            "name": "aqua",
            "price": 12000,
            "quantity": 1
        },
    ]
    ```
 