Dataset ModelsResource-RigNetv1 used in paper "RigNet: Neural Rigging for Articulated Characters" (ACM SIGGRAPH 2020). All models are originally downloaded from: https://www.models-resource.com/.

To better use these models for joint prediction, we removed overlapping and redundant joints, ensuring only one left at each position. We eliminated duplicates or near-duplicates whose IoU of volumes was more than 95%. We also manually verified that such re-meshed versions were filtered out. Under the guidance of an artist, we also verified that allcharacters have plausible skinning weights and deformations.

train/val/test_final.txt are the model-ID lists for all splits.

Thanks for your interests. Any questions please contact zhanxu@cs.umass.edu.