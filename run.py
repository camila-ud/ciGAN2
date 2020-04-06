from train import *

"experiment"
patch_size = 256
#learn_rate = 1e-4
batch_size = 8
l1_factor = 1200.0

def get_model():
    type_ = "lsgan"
    learn_rate = 1e-4
    opt = "rms"
    vgg = False
    inside = False    
    save_name = "{}_{}_{}_{:.1e}_end".format(type_,opt,int(vgg),learn_rate)
    
    ####### create model
    new_model = False 
    load_weights = False

    train_vgg = vgg
    load_vgg = vgg
    load_name = save_name

    model = CiGAN(save_name, load_name, patch_size, epochs,batch_size, 
              new_model, train_vgg=train_vgg, load_vgg=load_vgg,
              load_weights=load_weights,l1_factor = l1_factor, type = type_,
              save_model = False,inside = inside,learn_rate=learn_rate)
    model.build_model(batch_normalization=True)
    return model

def results(model, exp, id_ = 0):
    if exp == "val":  
        model.validate_model()
    elif exp == "syn":
        #2 its not cancer
        model.synthesis(id_)
    elif exp == "convert":
        print("Synthesize_dataset {} images".format(int(id_*8)))
        model.synthesize_dataset(id_)
    

if __name__ == '__main__':
    # only validating the model
    model = get_model()
    if '--val' in sys.argv:
        # do not save new model
        results(model,"val")
    elif '--syn' in sys.argv:
            # do not save new model
            test_id = sys.argv[2]
            print("sYn",test_id)
            results(model,"syn",test_id)
    elif '--convert' in sys.argv:
           results(model,"convert",4)
    else:
        print("Opt not valid, please select --val, --syn or --convert")
