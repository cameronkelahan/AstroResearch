import numpy as np
from keras.models import load_model
from sklearn import metrics
import matplotlib.pyplot as plt

# Plot heat map for given model; pass title and saveName
def plot(model, title, saveName):
    # Create the x and y axis values (0 - 1 stepping by .1)
    # xAxis = np.linspace(0, 1, num=11)
    # yAxis = np.linspace(0, 1, num=11)

    # Create the x and y axis values (0 - 1 stepping by .01)
    xAxis = np.linspace(0, 1, num=101)
    yAxis = np.linspace(0, 1, num=101)

    # The X data set to populate and predict probability
    predX = []
    for x in xAxis:
        for y in yAxis:
            predX.append([x, y])

    predProb = model.predict_proba(np.array(predX))
    predClass = model.predict(np.array(predX))

    # # Unused with the NN predict class
    # predMaser = predProb[:,1]

    # predMaser = predMaser.reshape(11, 11)
    predMaser = predProb.reshape(101, 101)
    predMaser = predMaser.transpose()


    plt.figure(figsize=(6.4, 4.8))
    plt.imshow(predMaser, origin='lower', extent=[0,1,0,1])
    plt.xticks(np.arange(.2, 1.1, step=0.2))  # Set label locations
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Predicted Prob of Maser', fontsize=16)
    plt.clim(vmin=0, vmax=1)
    cbar.set_clim(0,1)
    # plt.text(-10, 50, t, family='serif', ha='right', wrap=True)
    plt.title(title)
    plt.xlabel('L12',fontsize=16)
    plt.ylabel('Lx',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # Save as a PDF
    # plt.savefig(saveName, dpi=400, bbox_inches='tight', pad_inches=0.05)
    plt.show()
    plt.clf()
    plt.close()

############################# HEAT MAP OF NN  BASED ON UNWEIGHTED KNN DATASETS##########################################

###################### Load the models which were trained using the data from Unw KNN Dataset 1
model80Acc3Layer = load_model('./DataSetUnwKNN80+Acc/NeuralNetworkInfo/kerasModel3Layer_12_8_1.h5')
model80Acc4Layer = load_model('./DataSetUnwKNN80+Acc/NeuralNetworkInfo/kerasModel4Layer_8_12_4_1.h5')
model80Acc6Layer = load_model('./DataSetUnwKNN80+Acc/NeuralNetworkInfo/kerasModel6LayerEpoch550_10_10_20_50_17_1.h5')
model80Acc10Layer = load_model('./DataSetUnwKNN80+Acc/NeuralNetworkInfo/kerasModel10LayerEpoch550_10_10_20_25_30_50_17_12_8_1.h5')

################ 3 Layer NN Heat Map
plot(model80Acc3Layer, "3-Layer NN Maser Classification Probability Heat Map",
     './DataSetUnwKNN80+Acc/KerasProbHeatMap3Layer.pdf')

################# 4 Layer NN Heat Map
plot(model80Acc4Layer, "4-Layer NN Maser Classification Probability Heat Map",
     './DataSetUnwKNN80+Acc/KerasProbHeatMap4Layer.pdf')

################# 6 Layer NN Heat Map
plot(model80Acc6Layer, "6-Layer NN Maser Classification Probability Heat Map",
     './DataSetUnwKNN80+Acc/KerasProbHeatMap6Layer.pdf')

################# 10 Layer NN Heat Map
plot(model80Acc10Layer, "10-Layer NN Maser Classification Probability Heat Map",
     './DataSetUnwKNN80+Acc/KerasProbHeatMap10Layer.pdf')

############################# HEAT MAP OF NN BASED ON NN DATASETS#######################################################

# These models were trained by swapping out training sets every epoch
model82F1_3LayerV2 = load_model('./NNDataSelectionV2/3LayerModelF182/3LayerNNModel.h5')
model84F1_4LayerV2 = load_model('./NNDataSelectionV2/4LayerModelF184/4LayerNNModel.h5')

################# 3 Layer NN Heat Map V2
plot(model82F1_3LayerV2, "3-Layer NN Maser Classification Probability Heat Map",
     './NNDataSelectionV2/3LayerModelF182/heatMap3Layer.pdf')

################# 4 Layer NN Heat Map V2
plot(model84F1_4LayerV2, "4-Layer NN Maser Classification Probability Heat Map",
     './NNDataSelectionV2/4LayerModelF184/heatMap4Layer.pdf')
