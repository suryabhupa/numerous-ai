require 'nn'
require 'csvigo'
require 'torch'

local cmd = torch.CmdLine()
res = {}
cmd:text()
cmd:text('Train Numerai NN:')
cmd:text()
cmd:text('Options:')
cmd:option('-iterations', '10', 'number of training iterations')
cmd:text()

local opt = cmd:parse(arg)
print(opt)
ITERATIONS = opt.iterations

function get_data()
  train_data = csvigo.load({path="numerai_datasets/train_v_num.csv", mode="large"})
  train_targets = csvigo.load({path="numerai_datasets/train_v_targets.csv", mode="large"})
  val_data = csvigo.load({path="numerai_datasets/val_v_num.csv", mode="large"})
  val_targets = csvigo.load({path="numerai_datasets/val_v_targets.csv", mode="large"})
  test_data = csvigo.load({path="numerai_datasets/test_v_num.csv", mode="large"})
  test_ids = csvigo.load({path="numerai_datasets/test_v_ids.csv", mode="large"})

  return train_data, train_targets, val_data, val_targets, test_data, test_ids
end

function create_network()
  local mlp = nn.Sequential()
  inputs = 37
  outputs = 2
  HUs = inputs * 10
  mlp:add(nn.Dropout(0.1))
  mlp:add(nn.Linear(inputs, HUs))
  mlp:add(nn.Tanh())
  mlp:add(nn.Dropout(0.5))
  mlp:add(nn.Linear(HUs, outputs))

  return mlp
end

function make_dataset(data, targets)
  dataset = {}
  function dataset:size() return 41019 end
  c2 = 0
  for i=1,dataset:size() do
    local input = torch.Tensor(data[i])
    local tmp = targets[i][1]
    local output = torch.Tensor({1, 0})
    if tmp == "1" then
      output = torch.Tensor({0, 1})
    end
    dataset[i] = {input, output}
  end
  return dataset
end

function make_dataset2(data, targets)
  dataset = {}
  function dataset:size() return 41019 end
  c2 = 0
  for i=1,dataset:size() do
    local input = torch.Tensor(data[i])
    local tmp = targets[i][1]
    local output = torch.Tensor({1})
    if tmp == "1" then
      output = torch.Tensor({2})
    end
    dataset[i] = {input, output}
  end
  return dataset
end


function validate_and_report(mlp, val_data, val_targets)
  local count = 0
  for i=1,14019 do
    -- print(i)
    local input = torch.Tensor(val_data[i])
    local tmp = train_targets[i][1]
    local output = torch.Tensor({1, 0})
    if tmp == "1" then
      output = torch.Tensor({0, 1})
    end
    mlp_output = mlp:forward(input)
    if (mlp_output[1] >= 0.5 and output[1] == 1) or (mlp_output[1] < 0.5 and output[1] == 0) then
      count = count + 1
      -- print(mlp_output, output)
    end
  end
  return (count/14019)
end

function predict(mlp, test_data, test_ids)
  res = {}
  io.output("numerai_datasets/predictions.csv")
  function res:size() return 19461 end
  for i=1,res:size() do
    print(i)
    local input = torch.Tensor(test_data[i])
    local id = test_ids[i][1]
    mlp_output = mlp:forward(input)
    local pred = 0
    if (mlp_output[1] >= 0.5) then
      pred = 1
    end
    io.write(tonumber(id), ",", tonumber(pred), "\n")
  end
end

function main2()
  mlp = create_network()
  train_data, train_targets, val_data, val_targets, test_data, test_ids = get_data()
  train_dataset = make_dataset(train_data, train_targets)
  criterion = nn.CrossEntropyCriterion()

  for i=1,train_dataset:size() do
    output = 1
    if train_data[i][2][1] == 1 then
      output = 0
    end
    criterion:forward(mlp:forward(train_dataset[i][1]), output)
    mlp:zeroGradParameters()
    mlp:backward(train_dataset[i][1], criterion:backward(mlp.output, output))
    mlp:updateParameters(0.01)
  end

end

function main1()
  mlp = create_network()
  train_data, train_targets, val_data, val_targets, test_data, test_ids = get_data()
  train_dataset = make_dataset2(train_data, train_targets)

  criterion = nn.CrossEntropyCriterion()
  trainer = nn.StochasticGradient(mlp, criterion)
  trainer.learningRate = 0.1
  trainer.learningRateDecay = 0.98
  trainer.maxIteration = 100
  for i=1,1 do
    trainer:train(train_dataset)
    acc = validate_and_report(mlp, val_data, val_targets)
    print(acc)
  end
  predict(mlp, test_data, test_ids)
end

main1()
