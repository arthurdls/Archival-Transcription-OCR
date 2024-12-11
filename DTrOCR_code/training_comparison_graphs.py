pretraining = """
Epoch: 1 - Train loss: 1.6026719119759159, Train accuracy: 0.6882981224201977, Validation loss: 0.12469133686888033, Validation accuracy: 0.963640713268192
Epoch: 2 - Train loss: 0.080679693662835, Train accuracy: 0.9771281948985135, Validation loss: 0.02811532815252311, Validation accuracy: 0.9923483954014952
Epoch: 3 - Train loss: 0.02907656075259096, Train accuracy: 0.9918936731012994, Validation loss: 0.014938938015870224, Validation accuracy: 0.996133413848135
Epoch: 4 - Train loss: 0.01764573216314727, Train accuracy: 0.9951231447252378, Validation loss: 0.010629120319881738, Validation accuracy: 0.997293540413594
Epoch: 5 - Train loss: 0.012860538616703906, Train accuracy: 0.9964740618775534, Validation loss: 0.00915761380321727, Validation accuracy: 0.9977546929689268
Epoch: 6 - Train loss: 0.010046245543994494, Train accuracy: 0.9972534763575535, Validation loss: 0.00652193656119691, Validation accuracy: 0.9983628238650418
Epoch: 7 - Train loss: 0.008296610343449352, Train accuracy: 0.9977476208892943, Validation loss: 0.006217890190785593, Validation accuracy: 0.9985599561768475
"""
finetuning_no_frozen_weights = """
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
finetuning_frozen_weights = """
Epoch: 8 - Train loss: 6.22646785625384, Train accuracy: 0.2226138146578013, Validation loss: 3.9598548653237207, Validation accuracy: 0.4551516580942145
Epoch: 9 - Train loss: 5.573892165508062, Train accuracy: 0.26135150650348404, Validation loss: 3.6540159110242305, Validation accuracy: 0.4840087401501151
Epoch: 10 - Train loss: 5.265871860383483, Train accuracy: 0.2799373434627483, Validation loss: 3.4760384455646727, Validation accuracy: 0.5000048394574493
Epoch: 11 - Train loss: 5.0325812180890015, Train accuracy: 0.2941344441066908, Validation loss: 3.3549922694100127, Validation accuracy: 0.5100469002264973
Epoch: 12 - Train loss: 4.836098556230141, Train accuracy: 0.3062598489660646, Validation loss: 3.2742049834457023, Validation accuracy: 0.517959582017331
Epoch: 13 - Train loss: 4.660980401740011, Train accuracy: 0.3177113543640038, Validation loss: 3.2244191950976875, Validation accuracy: 0.5237855954598314
Epoch: 14 - Train loss: 4.500035436068054, Train accuracy: 0.32851831168963536, Validation loss: 3.149219482080779, Validation accuracy: 0.5299543978057977
Epoch: 15 - Train loss: 4.3494788755807505, Train accuracy: 0.33916172388529336, Validation loss: 3.115678733770146, Validation accuracy: 0.5346883946338955
Epoch: 16 - Train loss: 4.207597808305597, Train accuracy: 0.3501867685555643, Validation loss: 3.0769937108768053, Validation accuracy: 0.538404106727389
"""
no_pretraining_just_finetune_data = """
Epoch: 1 - Train loss: 7.121283507580365, Train accuracy: 0.11475603977616422, Validation loss: 6.240297780822736, Validation accuracy: 0.16493306535626753
Epoch: 2 - Train loss: 6.411277211387469, Train accuracy: 0.1352971008496418, Validation loss: 5.6840503403088505, Validation accuracy: 0.1915457256389171
Epoch: 3 - Train loss: 5.916760826619767, Train accuracy: 0.15829518801299924, Validation loss: 5.2754064659641395, Validation accuracy: 0.21771539314234672
Epoch: 4 - Train loss: 5.506626237262225, Train accuracy: 0.18090311624054523, Validation loss: 4.971040268727329, Validation accuracy: 0.24071647010290334
Epoch: 5 - Train loss: 5.146262606032129, Train accuracy: 0.20260629992023124, Validation loss: 4.718940246760035, Validation accuracy: 0.26246379175027024
Epoch: 6 - Train loss: 4.803373584258178, Train accuracy: 0.22503832130939466, Validation loss: 4.490853347936046, Validation accuracy: 0.282174278136477
Epoch: 7 - Train loss: 4.457910752812444, Train accuracy: 0.2499779684211216, Validation loss: 4.305907500825263, Validation accuracy: 0.30118135181990024
Epoch: 8 - Train loss: 4.145278642306934, Train accuracy: 0.27428082726385555, Validation loss: 4.176590617563481, Validation accuracy: 0.31652753797953326
Epoch: 9 - Train loss: 3.8589331010594554, Train accuracy: 0.2985019154787028, Validation loss: 4.086873094921484, Validation accuracy: 0.32912483770010814
Epoch: 10 - Train loss: 3.5903082298942675, Train accuracy: 0.3244855053239112, Validation loss: 4.016656373354083, Validation accuracy: 0.34056197661340976
Epoch: 11 - Train loss: 3.6680349641994607, Train accuracy: 0.34043847457856796, Validation loss: 2.825728737362518, Validation accuracy: 0.45145972669256645
Epoch: 12 - Train loss: 3.293502531132222, Train accuracy: 0.37418374207806054, Validation loss: 2.7463660500564036, Validation accuracy: 0.4666590237900067
Epoch: 13 - Train loss: 2.9878805818784655, Train accuracy: 0.4079041442448016, Validation loss: 2.6735077113982686, Validation accuracy: 0.482612899118786
Epoch: 14 - Train loss: 2.7098718016365515, Train accuracy: 0.44297290178360027, Validation loss: 2.620706985310532, Validation accuracy: 0.4969148943262603
Epoch: 15 - Train loss: 2.4524207440301633, Train accuracy: 0.47850816942252716, Validation loss: 2.5863262825324886, Validation accuracy: 0.5065832907481694
Epoch: 16 - Train loss: 2.2123478229559606, Train accuracy: 0.5146991098579206, Validation loss: 2.541386676721991, Validation accuracy: 0.521196889359707
"""
no_pretraining_just_finetune_data_256_max_positional_embedding = """
Epoch: 1 - Train loss: 6.818538792853836, Train accuracy: 0.1354007874812493, Validation loss: 5.632913623703335, Validation accuracy: 0.2096219040390713
Epoch: 2 - Train loss: 5.930244798895675, Train accuracy: 0.16805842225223427, Validation loss: 4.911902993244041, Validation accuracy: 0.25422986782674617
Epoch: 3 - Train loss: 5.3688010859282675, Train accuracy: 0.19964924136585518, Validation loss: 4.452309660431976, Validation accuracy: 0.2895611509198788
Epoch: 4 - Train loss: 4.91588050009066, Train accuracy: 0.22944766920567725, Validation loss: 4.0799216043119895, Validation accuracy: 0.32472739705719356
Epoch: 5 - Train loss: 4.500265880773034, Train accuracy: 0.2603034816445418, Validation loss: 3.8097912855891987, Validation accuracy: 0.3537838347107634
Epoch: 6 - Train loss: 4.150418844969505, Train accuracy: 0.2889325353409122, Validation loss: 3.6096202023640953, Validation accuracy: 0.3781157016265588
Epoch: 7 - Train loss: 3.8437854710064228, Train accuracy: 0.316352733770211, Validation loss: 3.456875678338358, Validation accuracy: 0.3993391048593969
Epoch: 8 - Train loss: 3.5609038053471855, Train accuracy: 0.34327675024250864, Validation loss: 3.35481026196934, Validation accuracy: 0.41615401709507566
Epoch: 9 - Train loss: 3.296281413605724, Train accuracy: 0.37097043006425967, Validation loss: 3.260367451328908, Validation accuracy: 0.43186073594115504
Epoch: 10 - Train loss: 3.0464738543520826, Train accuracy: 0.39916961836117704, Validation loss: 3.195881047209755, Validation accuracy: 0.4460077001674016
Epoch: 11 - Train loss: 2.8084411106561324, Train accuracy: 0.429450126387608, Validation loss: 3.1686435013045555, Validation accuracy: 0.45577938448659183
Epoch: 12 - Train loss: 2.5831034201262977, Train accuracy: 0.4604926208522496, Validation loss: 3.1309119338581506, Validation accuracy: 0.46670788103994354
Epoch: 13 - Train loss: 2.3659262721846575, Train accuracy: 0.49252451279639453, Validation loss: 3.133370515911464, Validation accuracy: 0.4749245777178191
Epoch: 14 - Train loss: 2.16081833029578, Train accuracy: 0.5243655532459325, Validation loss: 3.135256012404023, Validation accuracy: 0.48109519162112546
Epoch: 15 - Train loss: 1.9695968601553604, Train accuracy: 0.5548375472608661, Validation loss: 3.144253432380979, Validation accuracy: 0.4885970921341064
Epoch: 16 - Train loss: 1.7892229227137508, Train accuracy: 0.5852249916166578, Validation loss: 3.1668670744192404, Validation accuracy: 0.49367339012498185
"""

import matplotlib.pyplot as plt

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

finetuning_frozen_data = {
    "Epoch": list(range(8, 17)),
    "Train Accuracy": [
        0.2226138146578013, 0.26135150650348404, 0.2799373434627483, 0.2941344441066908,
        0.3062598489660646, 0.3177113543640038, 0.32851831168963536, 0.33916172388529336,
        0.3501867685555643
    ],
    "Validation Accuracy": [
        0.4551516580942145, 0.4840087401501151, 0.5000048394574493, 0.5100469002264973,
        0.517959582017331, 0.5237855954598314, 0.5299543978057977, 0.5346883946338955,
        0.538404106727389
    ]
}

no_pretrain_finetune_data = {
    "Epoch": list(range(1, 17)),
    "Train Accuracy": [
        0.11475603977616422, 0.1352971008496418, 0.15829518801299924, 0.18090311624054523,
        0.20260629992023124, 0.22503832130939466, 0.2499779684211216, 0.27428082726385555,
        0.2985019154787028, 0.3244855053239112, 0.34043847457856796, 0.37418374207806054,
        0.4079041442448016, 0.44297290178360027, 0.47850816942252716, 0.5146991098579206
    ],
    "Validation Accuracy": [
        0.16493306535626753, 0.1915457256389171, 0.21771539314234672, 0.24071647010290334,
        0.26246379175027024, 0.282174278136477, 0.30118135181990024, 0.31652753797953326,
        0.32912483770010814, 0.34056197661340976, 0.45145972669256645, 0.4666590237900067,
        0.482612899118786, 0.4969148943262603, 0.5065832907481694, 0.521196889359707
    ]
}

no_pretrain_finetune_256_data = {
    "Epoch": list(range(1, 17)),
    "Train Accuracy": [
        0.1354007874812493, 0.16805842225223427, 0.19964924136585518, 0.22944766920567725,
        0.2603034816445418, 0.2889325353409122, 0.316352733770211, 0.34327675024250864,
        0.37097043006425967, 0.39916961836117704, 0.429450126387608, 0.4604926208522496,
        0.49252451279639453, 0.5243655532459325, 0.5548375472608661, 0.5852249916166578
    ],
    "Validation Accuracy": [
        0.2096219040390713, 0.25422986782674617, 0.2895611509198788, 0.32472739705719356,
        0.3537838347107634, 0.3781157016265588, 0.3993391048593969, 0.41615401709507566,
        0.43186073594115504, 0.4460077001674016, 0.45577938448659183, 0.46670788103994354,
        0.4749245777178191, 0.48109519162112546, 0.4885970921341064, 0.49367339012498185
    ]
}

# Create the plot
plt.figure(figsize=(14, 8))

# Plot each dataset
plt.plot(finetuning_data["Epoch"], finetuning_data["Validation Accuracy"], label="Finetuning Validation Accuracy (No Frozen Weights)", linewidth=3)
plt.plot(finetuning_frozen_data["Epoch"], finetuning_frozen_data["Validation Accuracy"], label="Finetuning Validation Accuracy (Frozen Weights)")
plt.plot(no_pretrain_finetune_data["Epoch"], no_pretrain_finetune_data["Validation Accuracy"], label="No Pretraining Validation Accuracy")
plt.plot(no_pretrain_finetune_256_data["Epoch"], no_pretrain_finetune_256_data["Validation Accuracy"], label="No Pretraining Validation Accuracy (256 Max Positional Embedding)")

# Add labels, legend, and grid
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("Performance of Different Models", fontsize=18)
plt.legend(fontsize=15)
plt.grid(True)
plt.tight_layout()

# Save and show the plot
plt.savefig('DTrOCR_different_model_trials.png')
