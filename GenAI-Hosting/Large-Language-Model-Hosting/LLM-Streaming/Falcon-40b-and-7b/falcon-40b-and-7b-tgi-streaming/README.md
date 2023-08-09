## Building a chatbot with Amazon SageMaker streaming endpoint

You can deploy the [Falcon 40B Instruct](https://huggingface.co/tiiuae/falcon-40b-instruct) or [Falcon 7B Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) models on Amazon SageMaker endpoints using the [Text Generation Inference](https://huggingface.co/text-generation-inference) (TGI) container provided by Hugging Face. Follow the instructions in the corresponding notebooks to deploy the model to serve real-time inference requests. Once the endpoint is ready and tested successfully with streaming responses in the notebook. You can launch the streamlit application in the SageMaker Studio JupyterServer Terminal and access the chatbot UI from your browser. 

1. Open a System terminal in **SageMaker Studio**. On the top menu, choose **File**, then **New**, then **Terminal**.
2. Install the required python packages that are specified in the **requirements.txt** file.
```
$ pip install -r requirements.txt
```
3. Setup the environment variable with the endpoint name deployed in your account.
```
$ export endpoint_name=<the falcon endpoint name deployed in your account>
```
4. Launch the streamlit app from the `streamlit_chatbot_TGI.py` file, it will automatically update the endpoint names based on the environment variables.
```
$ streamlit run streamlit_chatbot_TGI.py --server.port 6006
```
To access the Streamlit UI, copy your SageMaker Studio url and replace `lab?` with `proxy/[PORT NUMBER]/`. Because we specified the server port to 6006, so the url should look like:
```
https://<domain ID>.studio.<region>.sagemaker.aws/jupyter/default/proxy/6006/
```
Replace the domain ID and region with the correct value in your account to access the chatbot UI. You can find some suggested prompts on the left-hand-side sidebar to get started.

