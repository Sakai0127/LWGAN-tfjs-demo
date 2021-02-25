let generator_lr;
tf.loadGraphModel('./models/G_LR/model.json').then(function (model) {
    generator_lr = model;
}).catch(
    () => console.log('Fail to load G_LR')
);
let generator_hr;
tf.loadGraphModel('./models/G_HR/model.json').then(function (model) {
    generator_hr = model;
}).catch(
    () => console.log('Fail to load G_HR')
);

let random_noise = tf.randomNormal([1, 256], 0, 1);
let style_noise = tf.randomNormal([1, 256], 0, 1);

function new_noise(){
    random_noise.dispose();
    random_noise = tf.randomNormal([1, 256], 0, 1);
}

function new_style(){
    style_noise.dispose();
    style_noise = tf.randomNormal([1, 256], 0, 1);
}

var hr_inp = generator_hr.inputs;
const sort_idx = {
    [String(hr_inp[0].shape[1])] : 0,
    [String(hr_inp[1].shape[1])] : 1,
    [String(hr_inp[2].shape[1])] : 2,
    [String(hr_inp[3].shape[1])] : 3
}

const noise_idx = sort_idx[128];

function sort_lr_outputs(x, y) {
    return sort_idx[String(x.shape[1])] - sort_idx[String(y.shape[1])]
}

function normalize(z) {
    var norm = tf.norm(z, 'euclidean', 1, true);
    var result = tf.div(z, norm);
    norm.dispose();
    return result
}

function to_int(hr_outputs) {
    hr_outputs = hr_outputs.squeeze().clipByValue(-1, 1).add(1).mul(127.5);
    var result = tf.cast(hr_outputs, 'int32');
    hr_outputs.dispose();
    return result
}

function sampling(){
    const result = tf.tidy(() => {
        var z = normalize(random_noise);
        var style_z = normalize(style_noise);
        var lr_out = generator_lr.predict(z).sort(sort_lr_outputs);
        var style_lr_output = generator_lr.predict(style_z).sort(sort_lr_outputs);
        var inp = [];
        for(var i=0; i<4; i++){
            if(i == noise_idx){
                inp.push(lr_out[i]);
            } else {
                inp.push(style_lr_output[i])
            }
        }
        var output_img = to_int(generator_hr.predict(inp))
        return output_img
    });
    return result
}