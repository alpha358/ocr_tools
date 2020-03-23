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
