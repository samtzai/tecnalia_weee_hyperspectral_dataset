from pytorch_lightning.callbacks import ModelCheckpoint




def get_callbacks_list(params, output_path):

    callbacks_list=[]

    # Callback used to save the best weights
    # Save intermediate weights (only when the weights that offer the best validation loss are wanted)
    if (params['train']['restore_best_weights']):
        checkpointer =ModelCheckpoint(
            monitor="val_loss",
            mode='min',
            dirpath=output_path,
            filename='weights',
            verbose=False,
            save_last=True
            )
        callbacks_list.append(checkpointer)

     
    return callbacks_list