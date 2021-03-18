import matplotlib.pyplot as plt



def plot_scatter_prediction()

    filetypeout = '.pdf'
    # timestamp = f"{round(time.time())}"

    if (os.path.exists(dir_save) == False):
        print("Error: Output directory does not exist")
        return

    try:
        # Training curve
        trainlossall_energy = hist.history['mean_absolute_error']
        testlossall_energy = hist.history['val_mean_absolute_error']
        outname = os.path.join(dir_save, "fit" + str(i) + "_loss" + filetypeout)
        plt.figure()
        plt.plot(np.arange(1, len(trainlossall_energy) + 1), trainlossall_energy, label='Training energy', color='c')
        plt.plot(np.array(range(1, len(testlossall_energy) + 1)) * epostep, testlossall_energy, label='Test energy',
                 color='b')
        plt.xlabel('Epochs')
        plt.ylabel('Mean absolute Error ' + "[" + unit_energy + "]")
        plt.title("Mean absolute Error vs. epochs")
        plt.legend(loc='upper right', fontsize='x-large')
        plt.savefig(outname)
        plt.close()
    except:
        print("Error: Could not plot loss curve")