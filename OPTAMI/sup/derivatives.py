import torch
from torch.autograd import Variable
import sys
sys.path.append('../Optami')
from Hyperfast_v2.OPTAMI.sup import tuple_to_vec as ttv


def third_derivative_vec(closure, params, vector):
    output = closure()
    grads = torch.autograd.grad(output, params, create_graph=True)
    dot = 0.
    for i in range(len(grads)):
        dot += grads[i].mul(vector[i]).sum()
    hvp = torch.autograd.grad(dot, params, create_graph=True)
    dot_hes = 0.
    for i in range(len(grads)):
        dot_hes += hvp[i].mul(vector[i]).sum()
    third_vp = torch.autograd.grad(dot_hes, params, retain_graph=False)
    hvp_det = []
    for pa in range(len(grads)):
        hvp_det.append(hvp[pa].detach())
    return third_vp, hvp_det

def flat_hessian(flat_grads, params, create_graph = False):
     full_hessian = []
     for l in range(flat_grads.size()[0]):
         temp_hess = torch.autograd.grad(flat_grads[l], params,
                                         retain_graph = True)
         # print(temp_hess)
         full_hessian.append(ttv.tuple_to_vector(temp_hess))
     return torch.stack(full_hessian)


def hessian(grads, params, create_graph = False):
    full_hessian = []
    flat_grads = grads.reshape(-1)  
    for l in range(flat_grads.size()[0]):
        temp_hess = torch.autograd.grad(flat_grads[l], params, 
                                        retain_graph = True,  
                                        create_graph = create_graph)
        #print(temp_hess)
        full_hessian.append(ttv.tuple_to_vector(temp_hess))
    return torch.stack(full_hessian).reshape(grads.shape + params.shape)
    
def third_derivative(hessian, grads, params, create_graph = False):
    full_third_derivative = []
    flat_grads = grads.reshape(-1)
    flat_hess = hessian.reshape((flat_grads.size()[0], flat_grads.size()[0]))
    for l in range(flat_hess.size()[0]):
        cur_hessian = []
        for j in range(flat_hess.size()[1]):
            temp_grad = torch.autograd.grad(flat_hess[l][j], params, 
                                            retain_graph = True,  
                                            create_graph = create_graph)
            cur_hessian.append(ttv.tuple_to_vector(temp_grad))
        cur_hessian = torch.stack(cur_hessian)
        full_third_derivative.append(cur_hessian)
    return torch.stack(full_third_derivative).reshape(hessian.shape + params.shape)  
    
def jacobian(y, x, create_graph = False):
    jac = []                             
    flat_y = y.reshape(-1)     
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):         
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)           
                                                                                                      
def hessian_2(y, x):  
    return jacobian(jacobian(y, x, create_graph=True), x, create_graph=True)        

def third_derivative_2(hess, x):
    return jacobian(hess, x, create_graph=True)   

def get_grad_hess_3der(model,
                       criterion,
                       grad_batch_size, 
                       hessian_batch_size, 
                       third_derivative_batch_size,
                       train_dataset,
                       use_reshape = False):
    parameters = list(model.parameters())
    num_parameters = len(parameters)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
                                               batch_size = grad_batch_size, 
                                               shuffle = False)
    for i, (grad_images, grad_labels) in enumerate(train_loader):
        break
    
    if use_reshape: 
        grad_images = Variable(grad_images.view(-1, 28 * 28))
        grad_labels = Variable(grad_labels).fmod(2)
        
    else:
        grad_images = Variable(grad_images)
        grad_labels = Variable(grad_labels)
        
    #print(grad_labels)
    
    loss_for_grads_part = criterion(model(grad_images), grad_labels)
    c = torch.autograd.grad(loss_for_grads_part, parameters, 
                            retain_graph = True,  
                            create_graph = True)
    
    hessian_images = grad_images[:hessian_batch_size]
    hessian_labels = grad_labels[:hessian_batch_size]
    loss_for_hess_part = criterion(model(hessian_images), hessian_labels)
    
    c_for_hess = torch.autograd.grad(loss_for_hess_part, parameters, 
                                     retain_graph = True,  
                                     create_graph = True)
    
    A = [hessian(c_for_hess[i], parameters[i], True) for i in range(num_parameters)]
    
    third_derivative_images = grad_images[:third_derivative_batch_size]
    third_derivative_labels = grad_labels[:third_derivative_batch_size]
    loss_for_third_der_part = criterion(model(third_derivative_images), third_derivative_labels)
    
    c_for_third_der = torch.autograd.grad(loss_for_third_der_part, parameters, 
                                          retain_graph = True,  
                                          create_graph = True)
    
    #return c, A, c
    A_for_third_der = [hessian(c_for_third_der[i], parameters[i], True) for i in range(num_parameters)]
    B = [third_derivative(A_for_third_der[i], 
                          c_for_third_der[i], 
                          parameters[i], True) for i in range(num_parameters)]
    
    return c, A, B
    
def get_grad_hess_3der_2(model,
                         criterion,
                         grad_batch_size, 
                         hessian_batch_size, 
                         third_derivative_batch_size,
                         train_dataset,
                         use_reshape = False):
    parameters = list(model.parameters())
    num_parameters = len(parameters)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
                                               batch_size = grad_batch_size, 
                                               shuffle = False)
    for i, (images, labels) in enumerate(train_loader):
        break
    
    if use_reshape: 
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels).fmod(2)
        
    else:
        images = Variable(images)
        labels = Variable(labels)
    
    third_derivative_images = images[:third_derivative_batch_size]
    third_derivative_labels = labels[:third_derivative_batch_size]
    loss_for_third_der_part = criterion(model(third_derivative_images), third_derivative_labels)
    
    c_for_third_der = torch.autograd.grad(loss_for_third_der_part, parameters, 
                                          retain_graph = True,  
                                          create_graph = True)
    
    A_for_third_der = [hessian(c_for_third_der[i], parameters[i], True) for i in range(num_parameters)]
    B = [third_derivative(A_for_third_der[i], 
                          c_for_third_der[i], 
                          parameters[i], True) for i in range(num_parameters)]
    
    hessian_images = images[third_derivative_batch_size:hessian_batch_size]
    hessian_labels = labels[third_derivative_batch_size:hessian_batch_size]
    loss_for_hess_part = criterion(model(hessian_images), hessian_labels)
    
    c_for_hess = torch.autograd.grad(loss_for_hess_part, parameters, 
                                     retain_graph = True,  
                                     create_graph = True)
    
    A = [(hessian(c_for_hess[i], parameters[i], True)*(hessian_batch_size - third_derivative_batch_size) + \
          A_for_third_der[i]*third_derivative_batch_size)/hessian_batch_size for i in range(num_parameters)]
    
    grad_images = images[hessian_batch_size:]
    grad_labels = labels[hessian_batch_size:]
    loss_for_grads_part = criterion(model(grad_images), grad_labels)
    c_for_grads = torch.autograd.grad(loss_for_grads_part, parameters, 
                                      retain_graph = True,  
                                      create_graph = True)
    
    c = [(c_for_hess[i]*(hessian_batch_size - third_derivative_batch_size) + \
          c_for_third_der[i]*third_derivative_batch_size + \
          c_for_grads[i]*(grad_batch_size - hessian_batch_size))/grad_batch_size for i in range(num_parameters)]
    
    return c, A, B
