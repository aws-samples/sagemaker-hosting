# [MME Load Testing](https://aws.amazon.com/blogs/machine-learning/best-practices-for-load-testing-amazon-sagemaker-real-time-inference-endpoints/)

- locust-distributed: traffic is evenly distributed across all models
- locust-top-n: 90% traffic goes to top n models (10 in this case)
- locust-single-model: 90% traffic goes to a single model
