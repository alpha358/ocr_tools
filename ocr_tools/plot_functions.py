from .utils import preds_to_integer
import matplotlib.pyplot as plt

def display_examples(model, test_loader, idx_to_text, n=20, p_tresh=0.5):
    '''
    Display test examples and model text with appropriate probs

    idx_to_text --- map indexes to char strings
    '''

    # read the model device
    DEVICE = next(model.parameters()).device

    # test loader should have small batch
    for x, y_gt in test_loader:

        # batch, character_class, sequence_element_number
        y_hat = model(x.to(DEVICE))


        batch_size = x.shape[0]
        eps = y_hat.shape[1] - 1 # the last class idx



        for n_example in range(batch_size):

            # Plot the probs distribution
            plt.pcolormesh(
                y_hat[n_example, :, :].detach().cpu().numpy(),
                edgecolors='k'
                )
            plt.yticks(
                        list(idx_to_text.keys()), 
                        list(idx_to_text.values())
                    )
            plt.ylabel('char_class')
            plt.xlabel('width')
            plt.show()

            preds, probs = preds_to_integer(
                y_hat[n_example, :, :].detach(), eps=eps, p_tresh=p_tresh
            )

            plt.imshow(x[n_example, :, :, :]/255) #gal reiktu pernormuoti pries input?
            #pdb.set_trace()
            plt.show()
            print([idx_to_text[idx] for idx in preds])
            print(probs)

            if n_example > n: break

        break


# ==============================================================================
# from IPython import display
# def plot_loss(LOSS, err):
#     '''
#     Display loss and Error rate
#     '''
#
#     display.clear_output(wait=True)
# #     plt.clf()
# #     try:
# #         fig.clear()
# #         plt.close(fig)
# #     except:
# #         pass
#
#     fig, ax = plt.subplots(1, 2, figsize=(20,10))
#     ax[0].plot(LOSS)
#     ax[0].set_yscale('log')
#     ax[0].set_title('LOSS')
#     ax[0].set_xlabel('step')
#
#
#     ax[1].plot(err)
#     ax[1].set_title('Mean error count')
#     ax[1].set_xlabel('step')
#
#     fig.show()
#     display.display(fig)
#
#     fig.clear()
#     plt.close(fig)
