"use client";
import Image from "next/image";
import { use, useState } from "react";

import axios, { formToJSON } from "axios";
import { useEffect } from "react";

import CssBaseline from "@mui/material/CssBaseline";
import Box from "@mui/material/Box";
import Container from "@mui/material/Container";
import TextField from "@mui/material/TextField";
import Stack from "@mui/material/Stack";
import Button from "@mui/material/Button";
import { LoadingButton } from "@mui/lab";
import { FormLabel } from "@mui/material";
import SendIcon from "@mui/icons-material/Send";

export default function Home() {
  const [username, setUsername] = useState("");
  const [lang, setLang] = useState("");
  const [symptom, setSymptom] = useState("");
  const [sentence, setSentence] = useState(null);
  const [sym_1, setSym_1] = useState(0);
  const [psym_1, setPsym_1] = useState(null);
  const [sentenceTwo, setSentenceTwo] = useState("");
  const [symptom2, setSymptom2] = useState("");
  const [sym_2, setSym_2] = useState(0);
  const [psym_2, setPsym_2] = useState(null);
  const [allSymptoms, setAllSymptoms] = useState([]);
  const [disease, setDisease] = useState(null);
  const [sentence3, setSentence3] = useState(null);
  const [posDisSym, setPosDisSym] = useState(null);
  const [i, setI] = useState(0);
  const [addQuestion, setAddQuestion] = useState(null);
  const [checkAddQuest, setCheckAddQuest] = useState(false);
  const [checkStart, setCheckStart] = useState(false);
  const [checkLoading, setCheckLoading] = useState(false);
  const [checkPs1, setCheckPs1] = useState(false);
  const [checkSuggest, setCheckSuggest] = useState(false);
  const [finalDiagnose, setFinalDiagnose] = useState("");
  const [checkLoad, setCheckLoad] = useState(false);

  useEffect(() => {
    console.log(sym_1, psym_1, sentenceTwo);
  }, [sym_1, psym_1, sentenceTwo]);

  const onAdd = () => {
    setCheckLoading(false);
    const data = { symptom: symptom, lang: lang, username: username };

    axios
      .post("http://127.0.0.1:5000/getsympfirst", data)
      .then(function (response) {
        setSym_1(response.data[0]);
        setPsym_1(response.data[1]);
        setSentenceTwo(response.data[2]);
      })
      .catch(function (error) {
        console.log(error);
        //Perform action based on error
      });
    setCheckPs1(true);
  };

  const onAdd2 = () => {
    setCheckLoading(true);
    const data = { symptom2: symptom2, lang: lang };
    let variable;
    axios
      .post("http://127.0.0.1:5000/getsympsecond", data)
      .then(function (response) {
        setSym_2(response.data[0]);
        setPsym_2(response.data[1]);
        variable = response.data[0];
      })
      .catch(function (error) {
        console.log(error);
        //Perform action based on error
      });
  };

  const onSubmit = async () => {
    setCheckLoading(true);
    const data = { username: username, lang: lang };
    console.log(data);
    await axios
      .post("http://127.0.0.1:5000/getuser", data)
      .then(function (response) {
        console.log(response);
        setSentence(response.data);
      })
      .catch(function (error) {
        console.log(error);
        //Perform action based on error
      });
    setCheckStart(true);
  };
  const suggestSymp = () => {
    setCheckSuggest(true);
    const pData = { psym_1: psym_1, psym_2: psym_2, lang: lang };
    console.log(sym_1, sym_2);
    if (sym_1 == 1 && sym_2 == 1) {
      axios
        .post("http://127.0.0.1:5000/setsympsall", pData)
        .then(function (response) {
          setAllSymptoms(response.data[0]);
          setDisease(response.data[1]);
          setSentence3(response.data[2]);
          setPosDisSym(response.data[3]);
          setAddQuestion(response.data[4]);
        })
        .catch(function (error) {
          console.log(error);
          //Perform action based on error
        });
    }
  };
  useEffect(() => {
    console.log(allSymptoms);
  });

  const yeaClick = (input) => {
    setAllSymptoms((prevArray) => [...prevArray, input]);
    setI(i + 1);

    if (input == posDisSym[posDisSym.length - 1]) {
      setCheckAddQuest(true);
    }
  };
  const nayClick = (input) => {
    if (input == posDisSym[posDisSym.length - 1]) {
      setCheckAddQuest(true);
    }
    setI(i + 1);
    console.log(allSymptoms);
  };

  const addSymp = () => {
    let checkSymptom = true;
    allSymptoms.filter((input) => {
      if (symptom == input) {
        checkSymptom = false;
      }
    });
    if (checkSymptom) {
      setAllSymptoms((prevArray) => [...prevArray, symptom]);
    }
    setSymptom("");
  };

  const clickDiagnose = () => {
    setCheckLoad(true);
    const data = { lang: lang, allSymptoms: allSymptoms };
    axios
      .post("http://127.0.0.1:5000/finaldiagnose", data)
      .then(function (response) {
        setFinalDiagnose(response.data);
      })
      .catch(function (error) {
        console.log(error);
        //Perform action based on error
      });
    setAddQuestion(null);
  };

  return (
    <div className="main">
      <CssBaseline />
      <Container>
        {!checkStart && (
          <div className="buttonGroup">
            <TextField
              id="outlined-basic"
              label="Username"
              variant="outlined"
              onChange={(e) => setUsername(e.target.value)}
            />
            <Stack spacing={2} direction="row" style={{ marginTop: 25 }}>
              <Button onClick={() => setLang("en")} variant="outlined">
                English
              </Button>
              <Button onClick={() => setLang("tr")} variant="outlined">
                Turkish
              </Button>
              <Button onClick={() => setLang("es")} variant="outlined">
                Espanol
              </Button>
            </Stack>

            <div style={{ marginTop: 20 }}>
              <Button variant="contained" color="success" onClick={onSubmit}>
                {lang == "tr" ? "Başla" : "Start"}
              </Button>
            </div>
            <LoadingButton
              size="large"
              style={{ marginTop: 10 }}
              loading={checkLoading}
            />
          </div>
        )}

        {sentence && !checkAddQuest && !checkPs1 && (
          <div
            style={{
              justifySelf: "center",
              flexDirection: "column",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              gap: 20,
              fontSize: 40,
            }}
          >
            <FormLabel
              style={{
                fontSize: 30,
              }}
            >
              {sentence}
            </FormLabel>

            <TextField
              style={{
                width: "30%",
              }}
              id="outlined-basic"
              label={lang == "tr" ? "İlk Semptom" : "First Symptom"}
              variant="outlined"
              onChange={(e) => setSymptom(e.target.value)}
            />

            <Button onClick={onAdd} variant="contained" color="success">
              {lang == "tr" ? "Semptom Ekle" : "Add Symptom"}
            </Button>
          </div>
        )}
        {psym_1 && !checkAddQuest && !checkLoading && (
          <div
            style={{
              justifySelf: "center",
              flexDirection: "column",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              gap: 20,
              fontSize: 40,
            }}
          >
            <FormLabel
              style={{
                fontSize: 30,
              }}
            >
              {sentenceTwo}
            </FormLabel>

            <TextField
              style={{
                width: "30%",
              }}
              id="outlined-basic"
              label={lang == "tr" ? "İkinci Semptom" : "Second Symptom"}
              variant="outlined"
              onChange={(e) => setSymptom2(e.target.value)}
            />

            <Button onClick={onAdd2} variant="contained" color="success">
              {lang == "tr" ? "Semptom Ekle" : "Add Symptom"}
            </Button>
          </div>
        )}

        {psym_2 && !checkAddQuest && !checkSuggest ? (
          <div>
            <Button
              variant="contained"
              onClick={suggestSymp}
              endIcon={<SendIcon />}
            >
              {lang == "tr" ? "Semptom Öner" : "Suggest Symptom"}
            </Button>
          </div>
        ) : (
          <LoadingButton
            size="large"
            style={{ marginTop: 10 }}
            loading={!disease && checkSuggest}
          />
        )}
        {disease && !checkAddQuest && (
          <div>
            {/* <label>{sentence3}</label> */}
            <FormLabel
              style={{
                fontSize: 30,
              }}
            >
              {sentence3}
            </FormLabel>
            <div>
              {/* <span>
                {posDisSym[i] == allSymptoms[0] ||
                posDisSym[i] == allSymptoms[1]
                  ? setI(i + 1)
                  : posDisSym[i]}
              </span> */}
              <FormLabel
                style={{
                  marginTop: 20,
                  fontSize: 26,
                  border: "3px solid lightblue",
                  margin: 10,
                  padding: 5,
                }}
              >
                {posDisSym[i] == allSymptoms[0] ||
                posDisSym[i] == allSymptoms[1]
                  ? setI(i + 1)
                  : posDisSym[i]}
              </FormLabel>
              <div style={{ gap: 10, padding: 10 }}>
                {/* <input
                  type="button"
                  value={lang == "tr" ? "Evet" : "Yes"}
                  onClick={() => yeaClick(posDisSym[i])}
                />
                <input
                  type="button"
                  value={lang == "tr" ? "Hayır" : "No"}
                  onClick={() => nayClick(posDisSym[i])}
                /> */}
                <Button
                  onClick={() => yeaClick(posDisSym[i])}
                  variant="outlined"
                  color="success"
                  style={{ margin: 10 }}
                >
                  {lang == "tr" ? "Evet" : "Yes"}
                </Button>
                <Button
                  onClick={() => nayClick(posDisSym[i])}
                  variant="outlined"
                  color="error"
                >
                  {lang == "tr" ? "Hayır" : "No"}
                </Button>
              </div>
            </div>
          </div>
        )}
        {checkAddQuest && addQuestion ? (
          <div>
            {/* <label>{addQuestion}</label> */}
            <div>
              <FormLabel
                style={{
                  fontSize: 30,
                }}
              >
                {addQuestion}
              </FormLabel>
            </div>
            {/* <input
              value={symptom}
              onChange={(e) => setSymptom(e.target.value)}
              className=""
              type="text"
              placeholder="Semptom"
            /> */}
            <TextField
              style={{
                margin: 10,
              }}
              id="outlined-basic"
              label={lang == "tr" ? "Semptom" : "Symptom"}
              variant="outlined"
              onChange={(e) => setSymptom(e.target.value)}
            />
            {/* <input
              value={lang == "tr" ? "Ekle" : "Add"}
              type="button"
              onClick={addSymp}
            /> */}
            <Button
              style={{ margin: 20 }}
              onClick={addSymp}
              variant="contained"
              color="success"
            >
              {lang == "tr" ? "Semptom Ekle" : "Add Symptom"}
            </Button>
            <div>
              {/* <input
                type="button"
                value={lang == "tr" ? "Son Teşhis" : "Final Diagnose"}
                onClick={clickDiagnose}
              /> */}
              <Button
                style={{ margin: 20 }}
                onClick={clickDiagnose}
                variant="contained"
                endIcon={<SendIcon />}
              >
                {lang == "tr" ? "Son Teşhis" : "Final Diagnose"}
              </Button>
            </div>
          </div>
        ) : (
          <LoadingButton
            color="error"
            size="large"
            style={{ marginTop: 10 }}
            loading={!finalDiagnose && checkLoad}
          />
        )}
        {finalDiagnose && (
          <div>
            <div>
              <FormLabel
                style={{
                  fontSize: 30,
                  color: "red",
                  marginBottom: 16,
                }}
              >
                {lang == "tr" ? "Son Teşhis" : "Final Diagnose"}
              </FormLabel>
            </div>
            <div>
              <FormLabel
                style={{
                  fontSize: 30,
                }}
              >
                {finalDiagnose}
              </FormLabel>
            </div>
          </div>
        )}
      </Container>
    </div>
  );
}
