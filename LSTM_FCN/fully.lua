require 'csvigo'
require 'caffe'
require 'itorch'
require 'image'
require 'nn'
require 'cutorch'
npy4th= require 'npy4th'
test_file='/nv/hp16/mrankawat3/caffe_future/data/CamVid/Images/Test/test.txt'
image_directory='/nv/hp16/mrankawat3/caffe_future/data/CamVid/Images/Test/'
file=torch.DiskFile(test_file,'r')
file:quiet()
mean_arr=npy4th.loadnpy('/nv/hp16/mrankawat3/caffe_future/examples/FCN-Alexnet/mean_camvid.npy')

function calc_result(confcounts)
classes={'Animal','Archway','Bicyclist','Bridge','Building','Car','CartLuggagePram','Child','Column_Pole','Fence','LaneMkgsDriv','LaneMkgsNonDriv','Misc_Text','MotorcycleScooter','OtherMoving','ParkingBlock','Pedestrian','Road','RoadShoulder','Sidewalk','SignSymbol','Sky','SUVPickupTruck','TrafficCone','TrafficLight','Train','Tree','Truck_Bus','Tunnel','VegetationMisc','Void','Wall'}
local num=32
accuracies=torch.DoubleTensor(num):fill(0)
gtj=torch.sum(confcounts,2)
resj=torch.sum(confcounts,1)
for i=1,num do
 gtj_ind=gtj[i][1] 
 resj_ind=resj[1][i]
 gtjresj=confcounts[i][i]
 if ((gtj_ind+resj_ind-gtjresj)>0)
  then accuracies[i]=100*(gtjresj)/(gtj_ind+resj_ind-gtjresj)
 else
  accuracies[i]=0
 end
 output=classes[i] .. " = " .. accuracies[i]
 print (output)  
end
mean_acc=torch.mean(accuracies)
print (mean_acc)
end
confcounts=torch.DoubleTensor(32,32):fill(0)
while true do
 local line = file:readString("*l")
 if (file:hasError()) then clearError() break
 else
   local im=image_directory .. line
   local num=32
   input= torch.FloatTensor(1,3,240,320)
   input[1]=image.load(im)
   input[1]=input[1]:index(1,torch.LongTensor{3,2,1})
   input[1]:add(input[1],-1,mean_arr:float())
   net=caffe.Net('/nv/hp16/mrankawat3/caffe_future/examples/FCN-Alexnet/deploy_camvid_32.prototxt','/nv/hp16/mrankawat3/caffe_future/examples/FCN-Alexnet/snapshot_camvid_modified/train_iter_80000.caffemodel','test')
   net:float()
   output=net:forward(input)
   y_out,ind_out=torch.max(output[1],1)
   ind_out:add(ind_out,-1)
   im_out='/nv/hp16/mrankawat3/caffe_future/data/CamVid/Output_Torch/' .. line
   ind1_out=ind_out
   ind1_out=ind1_out:float()
   ind1_out:mul(1/255)   
   image.save(im_out,ind1_out)
   gtim='/nv/hp16/mrankawat3/caffe_future/data/CamVid/GroundTruth/Test_Output/' .. line  
   gt=image.load(gtim)
   gt:mul(255):int()
   res=ind_out 
   maxlabel=torch.max(res)
   if(maxlabel>31)
     then print ('Result image has out of range value')
   end
   if (res:isSameSizeAs(gt:long()))
     then
   else
     print('Result image does not have same size')
   end
   res_num=torch.mul(res,num)
   add_num=torch.add(gt:double(),res_num:double())
   sumim=torch.add(add_num:double(),1)
   hs=torch.histc(sumim,num*num)
   confcounts=torch.add(confcounts,hs)     
 end
end
calc_result(confcounts) 


