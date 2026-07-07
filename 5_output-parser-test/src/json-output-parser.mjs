import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { JsonOutputParser } from '@langchain/core/output_parsers';

// 初始化模型
const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    temperature: 0,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
});

const parser = new JsonOutputParser();

// 这里我们不仅告诉模型“请以 JSON 返回”，还把 JsonOutputParser 的格式说明直接插入到 prompt 里。
// 这样可以让模型更严格地按照 parser 期望的输出规则来生成结果，避免出现额外说明、代码块、或者不规范的 JSON。
// question 中的自然语言要求是“我要什么”，而 parser.getFormatInstructions() 则是“怎么写才能让 parser.parse 成功"。
const question = `请介绍一下爱因斯坦的信息。请以 JSON 格式返回，包含以下字段：name（姓名）、birth_year（出生年份）、nationality（国籍）、major_achievements（主要成就，数组）、famous_theory（著名理论）。

${parser.getFormatInstructions()}`;

console.log('question:',question)
try {
    console.log("🤔 正在调用大模型（使用 JsonOutputParser）...\n");

    const response = await model.invoke(question);

    console.log("📤 模型原始响应:\n");
    console.log(response.content);

    const result = await parser.parse(response.content);

    console.log("✅ JsonOutputParser 自动解析的结果:\n");
    console.log(result);
    console.log(`姓名: ${result.name}`);
    console.log(`出生年份: ${result.birth_year}`);
    console.log(`国籍: ${result.nationality}`);
    console.log(`著名理论: ${result.famous_theory}`);
    console.log(`主要成就:`, result.major_achievements);

} catch (error) {
    console.error("❌ 错误:", error.message);
}
