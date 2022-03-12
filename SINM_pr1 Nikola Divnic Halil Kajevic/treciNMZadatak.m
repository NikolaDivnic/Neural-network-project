clc,clear,close all

%% unos
podaci = csvread ('Star_dataset.csv',1,0);
podaci = podaci';
podaci = podaci(1:1:9,:);
K1 = podaci(:,podaci(9,:) == 0);
K2 = podaci(:,podaci(9,:) == 1);

histogram(categorical(podaci(9,:),[0 1],{'K1','K2'}));

%% formiranje test i trening

prviGranica = round(0.8*length(K1));
drugiGrania = round(0.8*length(K2));

prviTrening = K1(1:8,1:prviGranica);
drugiTrening = K2(1:8,1:drugiGrania);
prviTest = K1(1:8,prviGranica+1:length(K1));
drugiTest = K2(1:8,drugiGrania+1:length(K2));
trening = [prviTrening,drugiTrening];
izlazTrening = [K1(9,1:prviGranica),K2(9,1:drugiGrania)];
test = [prviTest,drugiTest];
izlazTest = [K1(9,prviGranica+1:length(K1)),K2(9,drugiGrania+1:length(K2))];


%% glavno
najFunc = 0;
najArhi = 0;
arhitectures = [[20,20,20];[8,8,8];[11,8,14];[11,15,11]];
funkcije = ["poslin","tansig","purelin","logsig"];%purelin poslin tansig logsig
for i = 1:length(arhitectures)
   for fun = 1:length(funkcije)
       for  w =[2,3,4,5]
            for reg = [0.8,0.9,0.95]
                net = patternnet(arhitectures(i,:));
                net.divideFcn = 'divideind';
                net.divideParam.valInd = round(0.9*length(trening)):length(trening);
                net.divideParam.testInd = [];
                net.divideParam.trainInd = 1:round(0.9*length(trening));
                for j = 1:(length(net.layers)-1)  
                    net.layers{j}.transferFcn = convertStringsToChars(funkcije(fun));
                end
                net.trainParam.epochs = 1000;
                net.trainParam.min_grad = 1e-6;
                net.performParam.regularization = reg;
                net.trainParam.goal = 1e-10;
                net.trainParam.max_fail = 20;
                tezine = ones(1,length(trening));
                tezine(1,prviGranica+1:length(trening)) = w;
                [net,tr] = train(net,trening,izlazTrening,[],[],tezine);
                ulazValid = trening(:,round(0.9*length(trening)):length(trening));
                izlazValida = net(ulazValid);
                
                [e,cm] = confusion(izlazTrening(round(0.9*length(trening)):length(trening)),izlazValida);
                cm = cm';
                a = 1-e;
                p = cm(2,2) / (cm(2,2)+cm(2,1));
                r = cm(2,2) / (cm(2,2)+cm(1,2));
                f = 2*p*r/(p+r);
                
                if f>0.95*najFunc && a>najArhi 
                    najFunc = f;
                    najArhi = a;
                    arhitecture_best =  arhitectures(i,:);
                    wNaj = w;
                    najFuncAkt = convertStringsToChars(funkcije(fun));
                    najReg = reg;
                    najEpoha = tr.best_epoch;
                end
            end
        end
    end
end
%% optimalno
clc
net = patternnet(arhitecture_best);
net.divideFcn = '';
for i = 1:length(net.layers)-1
    net.layers{i}.transferFcn = najFuncAkt;
end
tezine = ones(1,length(trening));
tezine(1,prviGranica+1:length(trening)) = w;
net.performParam.regularization = najReg;
net.trainParam.goal = 1e-5;
net.trainParam.epochs = najEpoha;
net.trainParam.max_fail = 7;
net.trainParam.min_grad = 1e-6;
net.trainParam.showWindow = true;
net.trainParam.showCommandLine = false;
[net,tr] = train(net,trening,izlazTrening,[],[],tezine);
plotperform(tr);

%% test
testPredikcija = net(test);
figure
plotconfusion(izlazTest,testPredikcija);
[e,cm] = confusion(izlazTest,testPredikcija);
cm = cm';
rtest = cm(2,2) / (cm(2,2)+cm(1,2));
ptest = cm(2,2) / (cm(2,2)+cm(2,1));
predikcija_trening = net(trening);
figure
plotconfusion(izlazTrening,predikcija_trening);
[e,cm] = confusion(izlazTrening,predikcija_trening);
cm = cm';
rtrening = cm(2,2) / (cm(2,2)+cm(1,2));
ptrening = cm(2,2) / (cm(2,2)+cm(2,1));
