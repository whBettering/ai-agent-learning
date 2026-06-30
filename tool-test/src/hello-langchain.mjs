import { ChatOpenAI } from '@langchain/openai';
import 'dotenv/config';


const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME || "qwen3.7-plus",
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.BASE_URL || 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    },
});

const response = await model.invoke("介绍下自己");
console.log(response.content);