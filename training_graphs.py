pretraining = """
Epoch: 1 - Train loss: 1.6026719119759159, Train accuracy: 0.6882981224201977, Validation loss: 0.12469133686888033, Validation accuracy: 0.963640713268192
Epoch: 2 - Train loss: 0.080679693662835, Train accuracy: 0.9771281948985135, Validation loss: 0.02811532815252311, Validation accuracy: 0.9923483954014952
Epoch: 3 - Train loss: 0.02907656075259096, Train accuracy: 0.9918936731012994, Validation loss: 0.014938938015870224, Validation accuracy: 0.996133413848135
Epoch: 4 - Train loss: 0.01764573216314727, Train accuracy: 0.9951231447252378, Validation loss: 0.010629120319881738, Validation accuracy: 0.997293540413594
Epoch: 5 - Train loss: 0.012860538616703906, Train accuracy: 0.9964740618775534, Validation loss: 0.00915761380321727, Validation accuracy: 0.9977546929689268
Epoch: 6 - Train loss: 0.010046245543994494, Train accuracy: 0.9972534763575535, Validation loss: 0.00652193656119691, Validation accuracy: 0.9983628238650418
Epoch: 7 - Train loss: 0.008296610343449352, Train accuracy: 0.9977476208892943, Validation loss: 0.006217890190785593, Validation accuracy: 0.9985599561768475
"""
finetuning = """
Epoch: 8 - Train loss: 5.459342284019136, Train accuracy: 0.26819577602446343, Validation loss: 3.010589443269323, Validation accuracy: 0.533613738875818
Epoch: 9 - Train loss: 4.102101939981557, Train accuracy: 0.3544284361225116, Validation loss: 2.355405835521448, Validation accuracy: 0.5947206483235336
Epoch: 10 - Train loss: 3.2982882353261176, Train accuracy: 0.4235991558313953, Validation loss: 1.9469494588168046, Validation accuracy: 0.6410165535146597
Epoch: 11 - Train loss: 2.7141487875169243, Train accuracy: 0.4856557098559228, Validation loss: 1.6833267127950609, Validation accuracy: 0.6758420225472169
Epoch: 12 - Train loss: 2.258844633698331, Train accuracy: 0.5425846480928264, Validation loss: 1.4947614308395478, Validation accuracy: 0.7033261266439216
Epoch: 13 - Train loss: 1.898219825984479, Train accuracy: 0.5937671350364503, Validation loss: 1.367363719268808, Validation accuracy: 0.7244733516415717
Epoch: 14 - Train loss: 1.6027605144983401, Train accuracy: 0.640468156603487, Validation loss: 1.252951645489995, Validation accuracy: 0.7432184357214249
Epoch: 15 - Train loss: 1.3580524312580378, Train accuracy: 0.6830170358886766, Validation loss: 1.1776724512770105, Validation accuracy: 0.7567602725989215
Epoch: 16 - Train loss: 1.1554676080998618, Train accuracy: 0.7207521662159128, Validation loss: 1.1205517583645097, Validation accuracy: 0.767636039033103
"""

import matplotlib.pyplot as plt

# Data for pretraining
pretraining_data = {
    "Epoch": list(range(1, 8)),
    "Train Accuracy": [
        0.6882981224201977, 0.9771281948985135, 0.9918936731012994, 0.9951231447252378,
        0.9964740618775534, 0.9972534763575535, 0.9977476208892943
    ],
    "Validation Accuracy": [
        0.963640713268192, 0.9923483954014952, 0.996133413848135, 0.997293540413594,
        0.9977546929689268, 0.9983628238650418, 0.9985599561768475
    ]
}

# Data for finetuning
finetuning_data = {
    "Epoch": list(range(8, 17)),
    "Train Accuracy": [
        0.26819577602446343, 0.3544284361225116, 0.4235991558313953, 0.4856557098559228,
        0.5425846480928264, 0.5937671350364503, 0.640468156603487, 0.6830170358886766,
        0.7207521662159128
    ],
    "Validation Accuracy": [
        0.533613738875818, 0.5947206483235336, 0.6410165535146597, 0.6758420225472169,
        0.7033261266439216, 0.7244733516415717, 0.7432184357214249, 0.7567602725989215,
        0.767636039033103
    ]
}

# Create the plot
plt.figure(figsize=(14, 8))

# Plot pretraining data
plt.plot(pretraining_data["Epoch"], pretraining_data["Train Accuracy"], label="Pretraining Train Accuracy", linestyle='-', marker='^')
plt.plot(pretraining_data["Epoch"], pretraining_data["Validation Accuracy"], label="Pretraining Validation Accuracy", linestyle='-', marker='s')


# Add labels, legend, and grid
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("Pretraining Metrics", fontsize=18)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.savefig('DTrOCR_Training.png')

plt.clf

# Create the plot
plt.figure(figsize=(14, 8))

# Plot finetuning data
plt.plot(finetuning_data["Epoch"], finetuning_data["Train Accuracy"], label="Finetuning Train Accuracy", linestyle='-', marker='^')
plt.plot(finetuning_data["Epoch"], finetuning_data["Validation Accuracy"], label="Finetuning Validation Accuracy", linestyle='-', marker='s')

# Add labels, legend, and grid
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("Finetuning Metrics", fontsize=18)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()


plt.savefig('DTrOCR_Finetuning.png')