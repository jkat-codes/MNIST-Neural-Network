from train import test_prediction, load_saved_params, get_all_missed_predictions

W1, b1, W2, b2 = load_saved_params()

run = True

## Allows you to test the model on the test set 
# while run: 
#     index = int(input("Enter image number: ")) ## Defines image from test set

#     if index < 0: ## exits while loop when invalid index is entered
#         run = False
#         print("Exiting now...")
#         break

#     test_prediction(index, W1, b1, W2, b2)

## Will find 9 of the images that the model inaccurately labeled
# get_all_missed_predictions(W1, b1, W2, b2)

for i in range(0, 50): 
    test_prediction(i, W1, b1, W2, b2, plot=False)

