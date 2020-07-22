import numpy as np
from keras.models import load_model
from sklearn import metrics
import matplotlib.pyplot as plt

# ############################# HEAT MAP OF NN  BASED ON UNWEIGHTED KNN DATASETS##########################################
#
# ###################### Load the models which were trained using the data from Unw KNN Dataset 1
# model80Acc3Layer = load_model('./DataSetUnwKNN80+Acc/NeuralNetworkInfo/kerasModel3Layer_12_8_1.h5')
# model80Acc4Layer = load_model('./DataSetUnwKNN80+Acc/NeuralNetworkInfo/kerasModel4Layer_8_12_4_1.h5')
# model80Acc6Layer = load_model('./DataSetUnwKNN80+Acc/NeuralNetworkInfo/kerasModel6LayerEpoch550_10_10_20_50_17_1.h5')
# model80Acc10Layer = load_model('./DataSetUnwKNN80+Acc/NeuralNetworkInfo/kerasModel10LayerEpoch550_10_10_20_25_30_50_17_12_8_1.h5')
#
# ################ 3 Layer NN Heat Map
# # Create the x and y axis values (0 - 1 stepping by .1)
# # xAxis = np.linspace(0, 1, num=11)
# # yAxis = np.linspace(0, 1, num=11)
#
# # Create the x and y axis values (0 - 1 stepping by .01)
# xAxis = np.linspace(0, 1, num=101)
# yAxis = np.linspace(0, 1, num=101)
#
# # The X data set to populate and predict probability
# predX = []
# for x in xAxis:
#     for y in yAxis:
#         predX.append([x, y])
#
# # print("Type of PredX: ", type(predX))
# # print("Type of PredX[0]: ", type(predX[0]))
# # print("PredX: ", np.array(predX).shape)
#
# predProb = model80Acc3Layer.predict_proba(np.array(predX))
# predClass = model80Acc3Layer.predict(np.array(predX))
#
# # count = 0
# # for value in predProb:
# #     print("Pred Prob value = ", predProb[count])
# #     print("Predicted Class = ", predClass[count])
# #     count += 1
# # print("PredProb, predClass: ", [predProb, predClass])
#
# # # Unused with the NN predict class
# # predMaser = predProb[:,1]
#
# # predMaser = predMaser.reshape(11, 11)
# predMaser = predProb.reshape(101, 101)
# predMaser = predMaser.transpose()
#
#
# plt.figure(figsize=(6.4, 4.8))
# plt.imshow(predMaser, origin='lower', extent=[0,1,0,1])
# plt.xticks(np.arange(.2, 1.1, step=0.2))  # Set label locations
# cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=16)
# cbar.set_label('Weighted Predicted Prob of Maser', fontsize=16)
# plt.clim(vmin=0, vmax=1)
# cbar.set_clim(0,1)
# # plt.text(-10, 50, t, family='serif', ha='right', wrap=True)
# plt.title("3-Layer NN Maser Classification Probability Heat Map")
# plt.xlabel('L12',fontsize=16)
# plt.ylabel('Lx',fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# # Save as a PDF
# plt.savefig('./DataSetUnwKNN80+Acc/KerasProbHeatMap3Layer.pdf', dpi=400, bbox_inches='tight', pad_inches=0.05)
# plt.show()
# plt.clf()
# plt.close()
#
# #####################################
#
# ################# 4 Layer NN Heat Map
# # Create the x and y axis values (0 - 1 stepping by .1)
# # xAxis = np.linspace(0, 1, num=11)
# # yAxis = np.linspace(0, 1, num=11)
#
# # Create the x and y axis values (0 - 1 stepping by .01)
# xAxis = np.linspace(0, 1, num=101)
# yAxis = np.linspace(0, 1, num=101)
#
# # The X data set to populate and predict probability
# predX = []
# for x in xAxis:
#     for y in yAxis:
#         predX.append([x, y])
#
# # print("Type of PredX: ", type(predX))
# # print("Type of PredX[0]: ", type(predX[0]))
# # print("PredX: ", np.array(predX).shape)
#
# predProb = model80Acc4Layer.predict_proba(np.array(predX))
# predClass = model80Acc4Layer.predict(np.array(predX))
#
# # count = 0
# # for value in predProb:
# #     print("Pred Prob value = ", predProb[count])
# #     print("Predicted Class = ", predClass[count])
# #     count += 1
# # print("PredProb, predClass: ", [predProb, predClass])
#
# # # Unused with the NN predict class
# # predMaser = predProb[:,1]
#
# # predMaser = predMaser.reshape(11, 11)
# predMaser = predProb.reshape(101, 101)
# predMaser = predMaser.transpose()
#
#
# plt.figure(figsize=(6.4, 4.8))
# plt.imshow(predMaser, origin='lower', extent=[0,1,0,1])
# plt.xticks(np.arange(.2, 1.1, step=0.2))  # Set label locations
# cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=16)
# cbar.set_label('Weighted Predicted Prob of Maser', fontsize=16)
# plt.clim(vmin=0, vmax=1)
# cbar.set_clim(0,1)
# # plt.text(-10, 50, t, family='serif', ha='right', wrap=True)
# plt.title("4-Layer NN Maser Classification Probability Heat Map")
# plt.xlabel('L12',fontsize=16)
# plt.ylabel('Lx',fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# # Save as a PDF
# plt.savefig('./DataSetUnwKNN80+Acc/KerasProbHeatMap4Layer.pdf', dpi=400, bbox_inches='tight', pad_inches=0.05)
# plt.show()
# plt.clf()
# plt.close()
#
# #####################################
#
# ################# 6 Layer NN Heat Map
# # Create the x and y axis values (0 - 1 stepping by .1)
# # xAxis = np.linspace(0, 1, num=11)
# # yAxis = np.linspace(0, 1, num=11)
#
# # Create the x and y axis values (0 - 1 stepping by .01)
# xAxis = np.linspace(0, 1, num=101)
# yAxis = np.linspace(0, 1, num=101)
#
# # The X data set to populate and predict probability
# predX = []
# for x in xAxis:
#     for y in yAxis:
#         predX.append([x, y])
#
# predProb = model80Acc6Layer.predict_proba(np.array(predX))
# predClass = model80Acc6Layer.predict(np.array(predX))
#
# # count = 0
# # for value in predProb:
# #     print("Pred Prob value = ", predProb[count])
# #     print("Predicted Class = ", predClass[count])
# #     count += 1
#
# # # Unused in NN Predict class
# # predMaser = predProb[:,1]
#
# # predMaser = predMaser.reshape(11, 11)
# predMaser = predProb.reshape(101, 101)
# predMaser = predMaser.transpose()
#
# plt.figure(figsize=(6.4, 4.8))
# plt.imshow(predMaser, origin='lower', extent=[0,1,0,1])
# plt.xticks(np.arange(.2, 1.1, step=0.2))  # Set label locations
# cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=16)
# cbar.set_label('Weighted Predicted Prob of Maser', fontsize=16)
# plt.clim(vmin=0, vmax=1)
# cbar.set_clim(0,1)
# # plt.text(-10, 50, t, family='serif', ha='right', wrap=True)
# plt.title("6-Layer NN Maser Classification Probability Heat Map")
# plt.xlabel('L12',fontsize=16)
# plt.ylabel('Lx',fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.title("NN Maser Classification Probability")
# # Save as a PDF
# plt.savefig('./DataSetUnwKNN80+Acc/KerasProbHeatMap6Layer.pdf', dpi=400, bbox_inches='tight', pad_inches=0.05)
# plt.show()
# plt.clf()
# plt.close()
#
# ######################################
#
# ################# 10 Layer NN Heat Map
# # Create the x and y axis values (0 - 1 stepping by .1)
# # xAxis = np.linspace(0, 1, num=11)
# # yAxis = np.linspace(0, 1, num=11)
#
# # Create the x and y axis values (0 - 1 stepping by .01)
# xAxis = np.linspace(0, 1, num=101)
# yAxis = np.linspace(0, 1, num=101)
#
# # The X data set to populate and predict probability
# predX = []
# for x in xAxis:
#     for y in yAxis:
#         predX.append([x, y])
#
# # print("Type of PredX: ", type(predX))
# # print("Type of PredX[0]: ", type(predX[0]))
# # print("PredX: ", np.array(predX).shape)
#
# predProb = model80Acc10Layer.predict_proba(np.array(predX))
# predClass = model80Acc10Layer.predict(np.array(predX))
#
# # count = 0
# # for value in predProb:
# #     print("Pred Prob value = ", predProb[count])
# #     print("Predicted Class = ", predClass[count])
# #     count += 1
#
# # # Unused with NN predict class
# # predMaser = predProb[:,1]
#
# # predMaser = predMaser.reshape(11, 11)
# predMaser = predProb.reshape(101, 101)
# predMaser = predMaser.transpose()
#
# plt.figure(figsize=(6.4, 4.8))
# plt.imshow(predMaser, origin='lower', extent=[0,1,0,1])
# plt.xticks(np.arange(.2, 1.1, step=0.2))  # Set label locations
# cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=16)
# cbar.set_label('Weighted Predicted Prob of Maser', fontsize=16)
# plt.clim(vmin=0, vmax=1)
# cbar.set_clim(0,1)
# # plt.text(-10, 50, t, family='serif', ha='right', wrap=True)
# plt.title("10-Layer NN Maser Classification Probability Heat Map")
# plt.xlabel('L12',fontsize=16)
# plt.ylabel('Lx',fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.title("NN Maser Classification Probability")
# # Save as a PDF
# plt.savefig('./DataSetUnwKNN80+Acc/KerasProbHeatMap10Layer.pdf', dpi=400, bbox_inches='tight', pad_inches=0.05)
# plt.show()
# plt.clf()
# plt.close()

############################# HEAT MAP OF NN BASED ON NN DATASETS#######################################################

###################### Load the models which were trained using the data from Unw KNN Dataset 1
# model84F1_4Layer = load_model('./DataSetNN84F1/kerasModel4Layer_8_12_4_1.h5')
# model84F1_6Layer = load_model('./DataSetNN84F1/kerasModel6Layer_10_10_20_50_17_1.h5')
# model84F1_10Layer = load_model('./DataSetNN84f1/kerasModel10Layer_10_10_20_25_30_50_17_12_8_1.h5')

model82F1_3Layer = load_model('./NNDataSelectionV2/3LayerModelF182/3LayerNNModel.h5')
model84F1_4Layer = load_model('./NNDataSelectionV2/4LayerModelF184/4LayerNNModel.h5')

################# 3 Layer NN Heat Map
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

# print("Type of PredX: ", type(predX))
# print("Type of PredX[0]: ", type(predX[0]))
# print("PredX: ", np.array(predX).shape)

predProb = model82F1_3Layer.predict_proba(np.array(predX))
predClass = model82F1_3Layer.predict(np.array(predX))

# count = 0
# for value in predProb:
    # print("Pred Prob value = ", predProb[count])
    # print("Predicted Class = ", predClass[count])
    # count += 1
# print("PredProb, predClass: ", [predProb, predClass])

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
cbar.set_label('Weighted Predicted Prob of Maser', fontsize=16)
plt.clim(vmin=0, vmax=1)
cbar.set_clim(0,1)
# plt.text(-10, 50, t, family='serif', ha='right', wrap=True)
# plt.title("3-Layer NN Maser Classification Probability Heat Map")
plt.xlabel('L12',fontsize=16)
plt.ylabel('Lx',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# Save as a PDF
plt.savefig('./NNDataSelectionV2/3LayerModelF182/heatMap3Layer.pdf', dpi=400, bbox_inches='tight', pad_inches=0.05)
plt.show()
plt.clf()
plt.close()

#####################################

################# 4 Layer NN Heat Map
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

# print("Type of PredX: ", type(predX))
# print("Type of PredX[0]: ", type(predX[0]))
# print("PredX: ", np.array(predX).shape)

predProb = model84F1_4Layer.predict_proba(np.array(predX))
predClass = model84F1_4Layer.predict(np.array(predX))

# count = 0
# for value in predProb:
    # print("Pred Prob value = ", predProb[count])
    # print("Predicted Class = ", predClass[count])
    # count += 1
# print("PredProb, predClass: ", [predProb, predClass])

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
cbar.set_label('Weighted Predicted Prob of Maser', fontsize=16)
plt.clim(vmin=0, vmax=1)
cbar.set_clim(0,1)
# plt.text(-10, 50, t, family='serif', ha='right', wrap=True)
# plt.title("4-Layer NN Maser Classification Probability Heat Map")
plt.xlabel('L12',fontsize=16)
plt.ylabel('Lx',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# Save as a PDF
plt.savefig('./NNDataSelectionV2/4LayerModelF184/heatMap4Layer.pdf', dpi=400, bbox_inches='tight', pad_inches=0.05)
plt.show()
plt.clf()
plt.close()

# #####################################
#
# ################# 6 Layer NN Heat Map
# # Create the x and y axis values (0 - 1 stepping by .1)
# # xAxis = np.linspace(0, 1, num=11)
# # yAxis = np.linspace(0, 1, num=11)
#
# # Create the x and y axis values (0 - 1 stepping by .01)
# xAxis = np.linspace(0, 1, num=101)
# yAxis = np.linspace(0, 1, num=101)
#
# # The X data set to populate and predict probability
# predX = []
# for x in xAxis:
#     for y in yAxis:
#         predX.append([x, y])
#
# predProb = model84F1_6Layer.predict_proba(np.array(predX))
# predClass = model84F1_6Layer.predict(np.array(predX))
#
# # count = 0
# # for value in predProb:
#     # print("Pred Prob value = ", predProb[count])
#     # print("Predicted Class = ", predClass[count])
#     # count += 1
#
# # # Unused in NN Predict class
# # predMaser = predProb[:,1]
#
# # predMaser = predMaser.reshape(11, 11)
# predMaser = predProb.reshape(101, 101)
# predMaser = predMaser.transpose()
#
# plt.figure(figsize=(6.4, 4.8))
# plt.imshow(predMaser, origin='lower', extent=[0,1,0,1])
# plt.xticks(np.arange(.2, 1.1, step=0.2))  # Set label locations
# cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=16)
# cbar.set_label('Weighted Predicted Prob of Maser', fontsize=16)
# plt.clim(vmin=0, vmax=1)
# cbar.set_clim(0,1)
# # plt.text(-10, 50, t, family='serif', ha='right', wrap=True)
# plt.title("6-Layer NN Maser Classification Probability Heat Map")
# plt.xlabel('L12',fontsize=16)
# plt.ylabel('Lx',fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.title("NN Maser Classification Probability")
# # Save as a PDF
# plt.savefig('./DataSetNN84F1/KerasProbHeatMap6Layer.pdf', dpi=400, bbox_inches='tight', pad_inches=0.05)
# plt.show()
# plt.clf()
# plt.close()
#
# ######################################
#
# ################# 10 Layer NN Heat Map
# # Create the x and y axis values (0 - 1 stepping by .1)
# # xAxis = np.linspace(0, 1, num=11)
# # yAxis = np.linspace(0, 1, num=11)
#
# # Create the x and y axis values (0 - 1 stepping by .01)
# xAxis = np.linspace(0, 1, num=101)
# yAxis = np.linspace(0, 1, num=101)
#
# # The X data set to populate and predict probability
# predX = []
# for x in xAxis:
#     for y in yAxis:
#         predX.append([x, y])
#
# # print("Type of PredX: ", type(predX))
# # print("Type of PredX[0]: ", type(predX[0]))
# # print("PredX: ", np.array(predX).shape)
#
# predProb = model84F1_10Layer.predict_proba(np.array(predX))
# predClass = model84F1_10Layer.predict(np.array(predX))
#
# # count = 0
# # for value in predProb:
#     # print("Pred Prob value = ", predProb[count])
#     # print("Predicted Class = ", predClass[count])
#     # count += 1
#
# # # Unused with NN predict class
# # predMaser = predProb[:,1]
#
# # predMaser = predMaser.reshape(11, 11)
# predMaser = predProb.reshape(101, 101)
# predMaser = predMaser.transpose()
#
# plt.figure(figsize=(6.4, 4.8))
# plt.imshow(predMaser, origin='lower', extent=[0,1,0,1])
# plt.xticks(np.arange(.2, 1.1, step=0.2))  # Set label locations
# cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=16)
# cbar.set_label('Weighted Predicted Prob of Maser', fontsize=16)
# plt.clim(vmin=0, vmax=1)
# cbar.set_clim(0,1)
# # plt.text(-10, 50, t, family='serif', ha='right', wrap=True)
# plt.title("10-Layer NN Maser Classification Probability Heat Map")
# plt.xlabel('L12',fontsize=16)
# plt.ylabel('Lx',fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.title("NN Maser Classification Probability")
# # Save as a PDF
# plt.savefig('./DataSetNN84F1/KerasProbHeatMap10Layer.pdf', dpi=400, bbox_inches='tight', pad_inches=0.05)
# plt.show()
# plt.clf()
# plt.close()
