def display_examples(model, test_loader, n=20, p_tresh=0.001):
    '''
    Display test examples and model text with appropriate probs
    '''

    # read the model device
    DEVICE = next(model.parameters()).device

    # test loader should have small batch
    for x, y_gt in test_loader:

        # batch, character_class, sequence_element_number
        y_hat = model(x.to(DEVICE))

        n_iter = 0
        for n_example in range(batch_size):
            n_iter += 1

            preds, probs = preds_to_integer(y_hat[n_example, :, :], p_tresh=0.001)

            plt.imshow(x[n_example, :, :, :]/255) #gal reiktu pernormuoti pries input?
            #pdb.set_trace()
            plt.show()
            print([idx_to_text[idx] for idx in preds])
            print(probs)

            if n_iter > n: break

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
