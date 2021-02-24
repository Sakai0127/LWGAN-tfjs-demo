let canvas;
let context;
let image_data;
let lines = [];

function to_canvas(tensor){
    var array = tensor.arraySync();
    for(var x=0; x<512; x++){
        for(var y=0; y<512; y++){
            var idx = (x*512 + y)*4;
            image_data.data[idx] = array[x][y][0];
            image_data.data[idx+1] = array[x][y][1];
            image_data.data[idx+2] = array[x][y][2];
            image_data.data[idx+3] = 255;
        }
    }
    tensor.dispose();
    context.putImageData(image_data, 0, 0);
}

function generate() {
    $("#gif-anime").hide();
    lines[2].setOptions({dash:{animation:true}});
    to_canvas(sampling());
    lines[2].dash = false;
}
        
function openEditor(elem){
    $("#editor-title").text(`${elem.value} Editor`);
    var current;
    if(elem.value == "Style"){
        current = style_noise.arraySync();
        $(".rnd-style").css("visibility", "visible");
    } else {
        current = random_noise.arraySync();
        $(".rnd-style").css("visibility", "hidden");
    }
    for(var i=0; i<256; i++){
        $(`#latent-${i}`).val(current[0][i]);
    }
    $("#latent-editor").fadeIn();
}

function closeEditor(){
    var latent = [];
    for(var i=0; i<256; i++){
        latent.push(parseFloat($(`#latent-${i}`).val()));
    }
    if($("#editor-title").text() == 'Style Editor'){
        style_noise.dispose();
        style_noise = tf.tensor(latent, [1, 256]);
    } else {
        random_noise.dispose();
        random_noise = tf.tensor(latent, [1, 256]);
    }
    $("#latent-editor").fadeOut();
}

function set_random(elem){
    console.log(elem.value)
    if(elem.value == "Style"){
        style_noise.dispose();
        style_noise = tf.randomNormal([1, 256], 0, 1);
        var rnd = style_noise.arraySync();
        for(var i=0; i<256; i++){
            $(`#latent-${i}`).val(rnd[0][i]);
        }
    } else {
        random_noise.dispose();
        random_noise = tf.randomNormal([1, 256], 0, 1);
    }
}

$(function(){
    canvas = $('#result')[0];
    context = canvas.getContext('2d');
    image_data = context.getImageData(0, 0, canvas.width, canvas.height);
    context.fillStyle = "rgb(255, 255, 255)";
    context.fillRect(0, 0,canvas.width, canvas.height);

    $('.overview').on('click', function(){
        for(var line of lines){
            line.hide();
        };
        $('#overview-content').fadeIn();
    });

    $('#overview-close, #overview-bg').click(function(){
        $('#overview-content').fadeOut();
        for(var line of lines){
            line.show();
        };
    });

    $("#editor-close").click(function(){
        closeEditor();
    });
});
        
$(window).on('load', function(){
    lines.push(new LeaderLine($('#style-editor')[0], $('#start')[0], {size:7}));
    lines.push(new LeaderLine($('#noise-editor')[0], $('#start')[0], {size:7}));
    lines.push(new LeaderLine($('#start')[0], $('#output')[0], {size:7}));
    var idx = Math.floor(Math.random() * 5);
    // $("#gif-anime").attr("src", `videos/sample_${idx}.mp4`);
    for(var i=0; i<256; i++){
        $('<input>').attr({
            type:'number',
            id:`latent-${i}`,
            value:0.0,
            step:0.0001
        }).appendTo("#grid-form");
    };
});