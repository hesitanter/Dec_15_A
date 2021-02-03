classdef recurrent_layer_modified < nnet.layer.Layer
    properties (Learnable)
        Wx;
        Wh;
    end
        
    methods
        function layer = recurrent_layer_modified(numInputs,name)
            layer.NumInputs = numInputs;
            layer.Name = name;
            layer.Wx = ones(1,16)*0.5;
            layer.Wh = ones(3,1)*0.5;
            % Set layer description.
            layer.Description = "Recurrent layer of " + numInputs +  ... 
                " inputs";
        end
        
        function Z = predict(layer, X) % X: 16 row, 1 column
            wx = layer.Wx;
            wh = layer.Wh;

            a = wx(1,1:4)*X([1,5,9,13],1);
            b = a*wh(1,1) + wx(1,5:8)*X([2,6,10,14],1);
            c = b*wh(2,1) + wx(1,9:12)*X([3,7,11,15],1);
            d = c*wh(3,1) + wx(1,13:16)*X([4,8,12,16],1);
            Z = [a;b;c;d;];
        end
    end
end