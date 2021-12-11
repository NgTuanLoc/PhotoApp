import React, { useEffect, useState } from "react";
import axios from "axios";
import Loading from "../components/utils/Loading";
import AOS from "aos";
import "aos/dist/aos.css";
const Detection = () => {
  const [previewImageUrl, setPreviewImageUrl] = useState(false);
  const [imageFile, setImageFile] = useState(null);
  const [imagePrediction, setImagePrediction] = useState("");
  const [loading, setLoading] = useState(false);
  const [perform, setPerform] = useState(0);

  useEffect(() => {
    AOS.init({ duration: 500 });
  }, []);

  const generatePreviewImageUrl = (file, callback) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);

    reader.onloadend = (e) => callback(reader.result);
  };

  const handleChange = (e) => {
    const file = e.target.files[0];
    if (!file) {
      return;
    }
    setImageFile(file);
    generatePreviewImageUrl(file, (previewImageUrl) => {
      setPreviewImageUrl(previewImageUrl);
    });
  };

  const uploadHandler = async () => {
    const formData = new FormData();
    formData.append("file", imageFile, "img.png");
    let t0 = performance.now();
    try {
      setLoading(true);
      const res = await axios.post("http://localhost:5000/upload", formData);
      await setImagePrediction(res.data);
      let t1 = performance.now();
      setPerform(t1 - t0);
      setLoading(false);
    } catch (error) {
      alert(error);
    }
  };

  const saveImage = async () => {
    var canvas = await document.createElement("canvas");
    var ctx = await canvas.getContext("2d");
    var img = await document.getElementById("downloadImage");
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    var data = canvas.toDataURL("image/png");
    window.location.href = data;
    var a = document.createElement("a");
    a.href = data;
    a.download = "download.png";
    document.body.appendChild(a);
    a.click();
  };

  return (
    <div className="detection">
      <div className="detection__upload">
        <div
          className="detection__upload-title"
          data-aos="flip-left"
          data-aos-easing="ease-out-cubic"
          data-aos-duration="500"
        >
          Upload your image
        </div>
        <div className="detection__upload-action">
          <div className="detection__upload-img" data-aos="fade-right">
            {previewImageUrl ? (
              <img className="detection__img" alt="" src={previewImageUrl} />
            ) : (
              <input
                type="file"
                name="file"
                className="detection__input-img"
                onChange={handleChange}
              />
            )}
          </div>
          <div className="detection__result" data-aos="fade-left">
            {loading ? <Loading /> : ""}
            {imagePrediction ? (
              <img
                className="detection__img"
                alt=""
                id="downloadImage"
                src={imagePrediction}
              />
            ) : (
              <div className="text">Your prediction will receive here</div>
            )}
          </div>
        </div>
        <div className="detection__info">
          {imagePrediction && (
            <div className="detection__total-time">
              Timing to get result:{" "}
              <span>{Intl.NumberFormat().format(perform)}</span> seconds
            </div>
          )}
          {previewImageUrl && (
            <button
              className="detection__button-predict"
              onClick={uploadHandler}
            >
              {" "}
              Predict
            </button>
          )}
          {imagePrediction && (
            <button className="detection__button-download" onClick={saveImage}>
              {" "}
              Download
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default Detection;
