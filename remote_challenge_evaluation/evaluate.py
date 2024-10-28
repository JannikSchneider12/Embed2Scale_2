import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import ast

from Linear_Probing import SimpleEmbeddingDataset, create_dataloader, linear_probing_model, train


def evaluate(test_annotation_file_path, user_submission_file_path, phase_codename, **kwargs):

    print("Starting Evaluation.....")
    
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """

    embedding_dim = 2048
    num_classes = 2
    batch_size=32
    num_epochs = 10
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    

    output = {}

    if phase_codename == "dev":
        print("Evaluating for Dev Phase")

    elif phase_codename == 'test':
        print("Evaluating for Test Phase")

    # 1) create dataloaders
    train_dataloader, val_dataloader = create_dataloader(submission_df_path=user_submission_file_path, 
                                                         annotations_json_path=test_annotation_file_path,
                                                         batch_size=batch_size)

    # 2) create Linear Probing Model
    model = linear_probing_model(embedding_dim=embedding_dim)
    
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    # 3) train the model and evaluate
    final_train_loss, final_val_loss = train(model=model, 
                                             train_dataloader=train_dataloader, 
                                             val_dataloader=val_dataloader,
                                             device=device,
                                             optimizer=optimizer,
                                             loss_fn=loss_fn,
                                             num_epochs=num_epochs)


    output['result'] = [
    {
        'train_split': {
            'train_final_train_loss': final_train_loss,
            'train_final_val_loss': final_val_loss,
            'train_Metric3': 123,
            'train_Total': 123,
        }
    },
    {
        'test_split': {
            'test_final_train_loss': final_train_loss,
            'test_final_val_loss': final_val_loss,
            'test_Metric3': 123,
            'test_Total': 123,
        }
    }
]

    # old format
    '''
    output['result'] = [
            {
                'train_split': {
                    'final_train_loss': final_train_loss,
                    'final_val_loss': final_val_loss,
                    'Metric3': 123,
                    'Total': 123,
                }
            },
            {
                'test_split': {
                    'final_train_loss': final_train_loss,
                    'final_val_loss': final_val_loss,
                    'Metric3': 123,
                    'Total': 123,
                }
            }
        ]
    '''
    
    '''
    # To display the results in the result file
    if phase_codename == "dev":

        output["submission_result"] = output["result"][0]

    elif phase_codename == "test":

        output["submission_result"] = output["result"][1]

    # print(output['submission_result'])
    '''
    print(f"Completed evaluation for {phase_codename}")

    return output
