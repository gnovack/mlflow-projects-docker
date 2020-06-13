import mlflow

mlflow.projects.run(
    './celebs-cnn', 
    backend='local', 
    synchronous=False,
    parameters={'batch_size': 32, 'epochs': 5, 'convolutions': 2})

# mlflow.projects.run(
#     './celebs-cnn', 
#     backend='local', 
#     synchronous=False,
#     parameters={'batch_size': 32, 'epochs': 5, 'convolutions': 2})