classdef recurrent_layer_modified < nnet.layer.Layer
    properties (Learnable)
        Wx;
        Wh;
    end
        
    methods
        function layer = recurrent_layer(numInputs,name)
            layer.NumInputs = numInputs;
            layer.Name = name;
            layer.Wx = ones(1,8)*0.5;
            layer.Wh = ones(3,1)*0.5;
            % Set layer description.
            layer.Description = "Recurrent layer of " + numInputs +  ... 
                " inputs";
        end
        
        function Z = predict(layer, X) % X: 8 row, 1 column
            wx = layer.Wx;
            wh = layer.Wh;

            a = wx(1,1:2)*X([1,5],1);
            b = a*wh(1,1) + wx(1,3:4)*X([2,6],1);
            c = b*wh(2,1) + wx(1,5:6)*X([3,7],1);
            d = c*wh(3,1) + wx(1,7:8)*X([4,8],1);
            Z = [a;b;c;d;];
        end
    end
end