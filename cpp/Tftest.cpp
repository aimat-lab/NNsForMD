/*
 * Load model saved using tensorflow SavedModel format.
 *
 * To access the names of the input and output tensors of the model, you can use the `saved_model_cli` tool, inside python bin/ folder.
 *
 * The syntax is (for the default tag-set and signature): `saved_model_cli show --dir /path/to/saved_model_folder/ --tag_set serve --signature_def serving_default`
 *
 * > TODO Fix code to use other SignatureDefs other than the default one (It is also possible to define the name of the tensors via code while saving the model, but the documentation isn't so clear).
 *
 * More info at: https://stackoverflow.com/questions/58968918/accessing-input-and-output-tensors-of-a-tensorflow-2-0-savedmodel-via-the-c-api?noredirect=1#comment109422705_58968918
*/

#include <iostream>
#include <stdio.h>
#include "Model.h"
#include "Tensor.h"


int main() {
    printf("This tensorflow version: %s\n", TF_Version());
    

    Model model("SavedModel_v0");
    
    std::cout << "Operations of SavedModel" << std::endl;
    std::vector<std::string> result = model.get_operations();
    for(int i = 0; i < result.size();i++) {
        std::cout << result[i] << std::endl;
    }
    std::cout << std::endl;
    // run saved_model_cli show --dir C:\\Users\\Patrick\\source\\repos\\Tftest\\Tftest\\SavedModel_v0 --tag_set serve --signature_def serving_default

    Tensor input{ model, "serving_default_input_1" }; 
    Tensor prediction{ model, "StatefulPartitionedCall"};
   

    std::vector<float> data(36);
    std::iota(data.begin(), data.end(), 0);

    // Feed data to input tensor
    input.set_data(data, { 1,12, 3 });

    // Run and show predictions
    model.run(input, prediction);

    // Get tensor with predictions
    std::vector<float> predictions;
    predictions = prediction.Tensor::get_data<float>();

    std::cout << "Model prediction:" << std::endl;
    for(int i = 0; i < predictions.size();i++) {
        std::cout << predictions[i] << std::endl;
    }

}


