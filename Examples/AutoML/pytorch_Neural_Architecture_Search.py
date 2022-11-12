import torch.nn as nn
import torch
import torch.optim as optim
import optuna

def fc(inp_dim: int, output_dim: int, act:nn=nn.ReLU())->nn.Sequential:
    """Function to define a fully connected block

    Args:
        inp_dim (int): Input dimension of the layer
        output_dim ([int]): Output dimension of the layer
        act ([Pytorch Activation layer type], optional): [Desired Activation Layer for the FC Unit]. Defaults to nn.ReLU().

    Returns:
        Sequential Model: Fully connected layer block
    """
    linear = nn.Linear(inp_dim, output_dim)
    nn.init.xavier_uniform_(linear.weight)
    linear.bias.data.fill_(0)
    fc_out = nn.Sequential(linear, act)
    return fc_out 

def createmodel(inp_sizes: list, state_dim:int,action_dim:int, act:nn)->nn.Sequential:
    """Create Model: 

    Args:
        inp_sizes (int) : Number of Master Node data required  (Number of task lists available)
        state_dim (int): Number of Edge Node data required 
        action_dim (int): 
        act :
    Returns: 
        s_grid_len : list of size of state for corresponding all_task_list
    """
    list_model = []
    for i, inp_size in enumerate(inp_sizes):
        if i == 0:
            list_model.append(fc(state_dim, inp_size, act=act))
        else:
            list_model.append(fc(inp_sizes[i-1], inp_size, act=act))

    list_model.append(fc(inp_size, action_dim, act=nn.Softmax()))
        
    model = nn.Sequential(*list_model)
    return model

batch_size = 16
input_shape =3
out_shape= 10
epochs =20
data = torch.randn(batch_size,input_shape)
targets= torch.randint(0, 9, (batch_size,))

def objective(trial):
    n_layers = trial.suggest_int('n_layers', 1, 10)
    min = 8
    max = 512
    
    
    sub_list = [trial.suggest_int(str(i), min, max) for i in range(n_layers)]
    
    model = createmodel(sub_list, input_shape, out_shape, nn.ReLU())
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(data)
        outputs_num = torch.argmax(outputs, dim=1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    acc = 0
    accuracy = (sum([acc+1 for i in range(len(outputs_num)) if targets[i]==outputs_num[i]])/(batch_size))*100
    return accuracy

study = optuna.create_study(direction='maximize')
out = study.optimize(objective, n_trials=500)
print('Studying best trial : ',study.best_trial)