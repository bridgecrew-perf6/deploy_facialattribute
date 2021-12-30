
var moustachevalue = "";
var beardvalue = "";
            function ValidateFileUpload() {
        var fuData = document.getElementById('img');
        var FileUploadPath = fuData.value;

        if (fuData.files[0].length != 0){
               $("#ver").show(1000);
    $("#form1").show(1000);
        }
        else{
              $("#ver").hide();
    $("#form1").hide();
        }

//To check if user upload any file
        if (FileUploadPath == '') {
            alert("Please upload an image");

        } else {
            var Extension = FileUploadPath.substring(
                    FileUploadPath.lastIndexOf('.') + 1).toLowerCase();

//The file uploaded is an image

if (Extension == "png"  || Extension == "jpeg" || Extension == "jpg" || Extension == "jfif" || Extension == "gif" || Extension == "webp") {

// To Display
                if (fuData.files && fuData.files[0]) {
                    var reader = new FileReader();

                    reader.onload = function(e) {
                $('#upload').attr('src', e.target.result).width(410).height(446);
                    }

                    reader.readAsDataURL(fuData.files[0]);
                }

            } 

//The file upload is NOT an image
else {
                alert("Please upload image of type png, jpg, jpeg");

                document.getElementById('img').value=null;

            }
        }
    }


$(document).ready(function(){
    
    $("form")[0].reset();
  $("#next1").click(function(){
    $("#form1").hide();
    $("#form2").show(1000);
  });

    $("#previous1").click(function(){
    $("#form3").hide();
    $("#form2").hide();
    $("#form1").show(1000);
  });
  $("#next2").click(function(){
    $("#form1").hide();
    $("#form2").hide();
    $("#form3").show(1000);
  });
      $("#previous2").click(function(){
    $("#form3").hide();
    $("#form1").hide();
    $("#form2").show(1000);
  });
  $("#next3").click(function(){
    $("#form1").hide();
    $("#form2").hide();
    $("#form3").hide();
    $("#form4").show(1000);
  });
      $("#previous3").click(function(){
    $("#form4").hide();
    $("#form1").hide();
    $("#form2").hide();
    $("#form3").show(1000);
  });
        $("#next4").click(function(){
    $("#form1").hide();
    $("#form2").hide();
    $("#form3").hide();
    $("#form4").hide();
    $("#form5").show(1000);

  });
      $("#previous4").click(function(){
        $("#form5").hide();
    
    $("#form1").hide();
    $("#form2").hide();
    $("#form4").show(1000);
  });
        $("#next5").click(function(){
    $("#form1").hide();
    $("#form2").hide();
    $("#form3").hide();
    $("#form4").hide();
    $("#form5").hide();
    $("#form6").show(1000);

  });
      $("#previous5").click(function(){
        
        $("#form6").hide();
    $("#form4").hide();
    $("#form3").hide();
    $("#form1").hide();
    $("#form2").hide();
    $("#form5").show(1000);
  });
              $("#next6").click(function(){
    $("#form1").hide();
    $("#form2").hide();
    $("#form3").hide();
    $("#form4").hide();
    $("#form5").hide();
    $("#form6").hide();
    $("#form7").show(1000);

  });
      $("#previous6").click(function(){
        
        $("#form7").hide();
    $("#form4").hide();
    $("#form3").hide();
    $("#form1").hide();
    $("#form2").hide();
    $("#form5").hide();
    $("#form6").show(1000);
  });
    $("#next7").click(function(){
    $("#form1").hide();
    $("#form2").hide();
    $("#form3").hide();
    $("#form4").hide();
    $("#form5").hide();
    $("#form6").hide();
    $("#form7").hide();
    $("#form8").show(1000);

  });
      $("#previous7").click(function(){
    $("#form8").hide();    
    $("#form6").hide();
    $("#form4").hide();
    $("#form3").hide();
    $("#form1").hide();
    $("#form2").hide();
    $("#form5").hide();
    $("#form7").show(1000);
  });
    $("#next8").click(function(){
    $("#form1").hide();
    $("#form2").hide();
    $("#form3").hide();
    $("#form4").hide();
    $("#form5").hide();
    $("#form6").hide();
    $("#form7").hide();
    $("#form8").hide();
    $("#form81").show(1000);

  });
      $("#previous8").click(function(){
    $("#form9").hide();   
    $("#form7").hide();    
    $("#form6").hide();
    $("#form4").hide();
    $("#form3").hide();
    $("#form1").hide();
    $("#form2").hide();
    $("#form5").hide();
    $("#form81").show(1000);
  });
          $("#next81").click(function(){
    $("#form1").hide();
    $("#form2").hide();
    $("#form3").hide();
    $("#form4").hide();
    $("#form5").hide();
    $("#form6").hide();
    $("#form7").hide();
    $("#form8").hide();
    $("#form81").hide();
    $("#form9").show(1000);

  });
      $("#previous71").click(function(){
    $("#form81").hide();
    $("#form9").hide();   
    $("#form7").hide();    
    $("#form6").hide();
    $("#form4").hide();
    $("#form3").hide();
    $("#form1").hide();
    $("#form2").hide();
    $("#form5").hide();
    $("#form8").show(1000);
  });
    $("#next9").click(function(){
    $("#form1").hide();
    $("#form2").hide();
    $("#form3").hide();
    $("#form4").hide();
    $("#form5").hide();
    $("#form6").hide();
    $("#form7").hide();
    $("#form8").hide();
    $("#form9").hide();
    $("#form10").show(1000);

  });
      $("#previous9").click(function(){
    $("#form10").hide();   
    $("#form7").hide();    
    $("#form6").hide();
    $("#form4").hide();
    $("#form3").hide();
    $("#form1").hide();
    $("#form2").hide();
    $("#form5").hide();
    $("#form8").hide();
    $("#form9").show(1000);
  });
          $("#next10").click(function(){
    $("#form1").hide();
    $("#form2").hide();
    $("#form3").hide();
    $("#form4").hide();
    $("#form5").hide();
    $("#form6").hide();
    $("#form7").hide();
    $("#form8").hide();
    $("#form9").hide();
    $("#form10").hide();
    $("#form11").show(1000);

  });
      $("#previous10").click(function(){
    $("#form11").hide();
    $("#form9").hide();   
    $("#form7").hide();    
    $("#form6").hide();
    $("#form4").hide();
    $("#form3").hide();
    $("#form1").hide();
    $("#form2").hide();
    $("#form5").hide();
    $("#form8").hide();
    $("#form10").show(1000);
  });
                $("#next11").click(function(){
    $("#form1").hide();
    $("#form2").hide();
    $("#form3").hide();
    $("#form4").hide();
    $("#form5").hide();
    $("#form6").hide();
    $("#form7").hide();
    $("#form8").hide();
    $("#form9").hide();
    $("#form10").hide();
    $("#form11").hide();
    $("#form12").show(1000);

  });
      $("#previous11").click(function(){
    $("#form12").hide();
    $("#form10").hide();
    $("#form9").hide();   
    $("#form7").hide();    
    $("#form6").hide();
    $("#form4").hide();
    $("#form3").hide();
    $("#form1").hide();
    $("#form2").hide();
    $("#form5").hide();
    $("#form8").hide();
    $("#form11").show(1000);
  });



});

function validatevalue(){
    var elements = document.getElementsByClassName("valuecheck");
    if(elements == null){
        alert("the form does not fill completely");
        return false;
    }
    else
    {
        alert("the form does not fill completely");
        return true;
    }

}

