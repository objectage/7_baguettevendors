const OpenAI = require('openai');
const { Configuration, OpenAIApi } = OpenAI;

const express = require('express');
const bodyParser = require('body-parser')
const cors = require('cors');
const app = express();
const port = 3001;

const configuration = new Configuration({
    organization: "org-ut8wbT8TDqt3BIVauUMZgoRS",
    apiKey: "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",
});
const openai = new OpenAIApi(configuration);

app.use(bodyParser.json());
app.use(cors());

app.post('/' , async (req, res) => {
    const { message } = req.body;
    const response = await openai.createCompletion({
        model: "text-davinci-003",
        prompt: `provide a solution for a farmer in karnataka who has the following problem. 
        Consider the climate and geographic elements of the region if needed. Simplify.
        problem :${message}`,
        max_tokens: 1000,
        temperature: 0.6,
      });
      console.log(response.data)
        if(response.data.choices[0].text){
                res.json({message: response.data.choices[0].text})
        }      
});

app.listen(port, () => {
    console.log('App listening');
});