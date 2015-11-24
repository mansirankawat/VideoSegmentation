require 'rnn'
require 'caffe'
require 'itorch'
require 'image'
require 'nn'
require 'cutorch'
npy4th= require 'npy4th'
--train_file='/nv/hp16/mrankawat3/data/CamVid/Images/Train/train.txt'
--image_directory='/nv/hp16/mrankawat3/data/CamVid/Images/Train/'
--file=torch.DiskFile(train_file,'r')
--file:quiet()
--mean_arr=npy4th.loadnpy('/nv/hp16/mrankawat3/caffe_future/examples/FCN-Alexnet/mean_camvid_train.npy')
--input=torch.FloatTensor(367,3,240,320)
--local i=1
--while true do
-- local line = file:readString("*l")
-- if (file:hasError()) then file:clearError() break
-- else
--   local im=image_directory .. line
--   input[i]=image.load(im)
--   input[i]=input[i]:index(1,torch.LongTensor{3,2,1})
--   input[i]:add(input[i],-1,mean_arr:float())
--   i=i+1
-- end
--end
input=torch.FloatTensor(367,240*320)
train_file='/nv/hp16/mrankawat3/data/CamVid/Output_Train_new/train.txt'
image_directory='/nv/hp16/mrankawat3/data/CamVid/Output_Train_new/'
file=torch.DiskFile(train_file,'r')
file:quiet()
local count=1
while true do
 local line = file:readString("*l")
 if (file:hasError()) then file:clearError() break
 else
   local im=image_directory .. line
   input[count]=torch.reshape(image.load(im),240*320,1)
   count=count+1
 end
end

print ("Reading Training Images Complete")
target=torch.FloatTensor(367,240*320)
train_labelfile='/nv/hp16/mrankawat3/data/CamVid/GroundTruth/Train/Output/label.txt'
image_directory_label='/nv/hp16/mrankawat3/data/CamVid/GroundTruth/Train/Output/'
file1=torch.DiskFile(train_labelfile,'r')
file1:quiet()
local count=1
while true do
 local line = file1:readString("*l")
 if (file1:hasError()) then file1:clearError() break
 else
   local im=image_directory_label .. line
   target[count]=torch.reshape(image.load(im),240*320,1)
   count=count+1
 end
end
print ("Reading Label Images Complete")


--LSTM

local lstm=nn.Sequential()
lstm:add(nn.LSTM(1,1,30))
lstm:add(nn.LogSoftMax())
lstm:training()
mlp=nn.ParallelTable()
for i=1,240*320 do
	mlp:add(lstm)
end
module=nn.Sequential()
mlp:training()
--net=caffe.Net('/nv/hp16/mrankawat3/caffe_future/examples/FCN-Alexnet/deploy_camvid_32.prototxt','/nv/hp16/mrankawat3/data/snapshot_camvid_modified/train_iter_80000.caffemodel','test')
--net:float()
--module:add(net)
module:add(mlp)
module:training()
final=nn.Sequencer(module)
final:training()

local criterion=nn.ClassNLLCriterion()
state = {
 learningRate = 0.1,
 momentum = 0.9
}

w,dw = final:getParameters()
w:uniform(-0.08,0.08)
feval = function(w_new)
	if w~=w_new then
		w:copy(w_new)
	end
	idx = (idx or 0)+1
	if idx > (#input)[1] then idx = 1 end
	local sample = input[idx]
	local label = target[idx]
        dw:zero()
        local loss = criterion:forward(final:forward(sample),label)
        final:backward(sample,criterion:backward(final.output,label))
        return loss,dw
end
for i=1,10000 do
	current_loss=0
   	for j=1,(#input)[1] do
		_,fs = optim.sgd(feval,w,state)
		current_loss=current_loss+fs[1]
	end
	current_loss=current_loss/(#input)[1]
	print ('Current loss= ' .. current_loss)
end



























































































































































































































































