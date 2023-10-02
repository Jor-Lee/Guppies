import numpy as np
import torch 

def UseYOLO(letter_model, number_model, image_file, probability_threshold=0.5, verbose=False):
    relevant_characters =  {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'B',10:'F',11:'G',12:'K',13:'N',14:'O',15:'P',16:'R',17:'S',18:'V',19:'W',20:'Y', 21:''}
    
    results_let = letter_model.predict(image_file)[0]
    results_num = number_model.predict(image_file)[0]

    boxes, pred_vec, probs, letter_idx = analyse_results(results_let.boxes.data,results_num.boxes.data, probability_threshold)
    characters = []
    for vec in pred_vec:
        characters.append(relevant_characters[vec])

    return characters, boxes, probs, letter_idx


def analyse_results(letter_data, number_data, prob_threshold):

    #first bring together the letters and numbers.

    letter_data = letter_data.detach().clone()
    letter_data[:,-1] = letter_data[:,-1] + 9 #add 9 to the class of the letters to make them different from the numbers

    no_letters = len(letter_data)
    data = torch.cat((letter_data, number_data), dim=0) #combine the two tensors
    idx_data = torch.arange(data.shape[0], dtype=torch.int) #create an index tensor

    #discard characters with low probability
    keep = data[:,-2] > prob_threshold
    data = data[keep] 
    idx_data = idx_data[keep]

    #sort by x coordinate to get the right order
    orde = torch.argsort(data[:,0],dim=0)
    data = data[orde] 
    idx_data = idx_data[orde]

    
    if data.shape[0] > 0:
        #discard any characters that are too short?
        height_threshold = (data[:,3] - data[:,1]).max() * 0.5
        
        keep = data[:,3] - data[:,1] > height_threshold
        data = data[keep]
        idx_data = idx_data[keep] 

        data, idx_data = discard_overlapping_boxes(data, idx_data) #discard overlapping boxes
        
    probs = data[:,-2].numpy()
    classes = data[:,-1].to(torch.int).numpy()

    boxes = data[:,:-2].numpy().astype(int)
    
    letter_idxs = torch.where(idx_data < no_letters)[0]
    return boxes, classes, probs, letter_idxs
    

def discard_overlapping_boxes(data, idx_data):
    #if the model predicted two characters that overlap alot, then remove the smaller one?

    intersections = np.array([len(list(range(max(int(data[i,0]), int(data[i+1,0])), min(int(data[i,2]), int(data[i+1,2]))+1))) for i in range(data.shape[0]-1)])#get the x intersection between each box
    area = (data[:,2]-data[:,0])*(data[:,3]-data[:,1])

    x_ranges = data[:,2]-data[:,0]

    x_ranges = np.array([torch.min(x_ranges[i],x_ranges[i+1]) for i in range(x_ranges.shape[0]-1)])

    intersections = intersections/x_ranges

    problemos = np.where(intersections>0.3)[0] #if 0.3 of a letter 

    for n in range(len(problemos)):
        i = problemos[n]
        if area[i] < area[i+1]:
            data = np.delete(data, i, axis=0)
            idx_data = np.delete(idx_data, i, axis=0)
        else:
            data = np.delete(data, i+1, axis=0)
            idx_data = np.delete(idx_data, i+1, axis=0)
        problemos = problemos - 1

    return data, idx_data