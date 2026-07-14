import 'dotenv/config';
import { MultiServerMCPClient } from '@langchain/mcp-adapters';
import { ChatOpenAI } from '@langchain/openai';
import chalk from 'chalk';
import { HumanMessage, ToolMessage } from '@langchain/core/messages';
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';
import { RunnableSequence, RunnableLambda, RunnableBranch, RunnablePassthrough } from '@langchain/core/runnables';

const model = new ChatOpenAI({ 
    modelName: "qwen-plus",
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
});

const mcpClient = new MultiServerMCPClient({
    mcpServers: {
        "amap-maps-streamableHTTP": {
            "url": "https://mcp.amap.com/mcp?key=" + process.env.AMAP_MAPS_API_KEY
        },
        "chrome-devtools": {
            "command": "npx",
            "args": [
                "-y",
                "chrome-devtools-mcp@latest"
            ]
        },
    }
});

const tools = await mcpClient.getTools();
const modelWithTools = model.bindTools(tools);

const prompt = ChatPromptTemplate.fromMessages([
    ["system", "你是一个可以调用 MCP 工具的智能助手。"],
    new MessagesPlaceholder("messages"),
]);

const llmChain = prompt.pipe(modelWithTools);

// 1. 定义处理工具调用的逻辑 (封装为 Runnable)
const toolExecutor = new RunnableLambda({
    func: async (input) => {
        const { response, tools } = input;
        const toolResults = [];

        for (const toolCall of response.tool_calls ?? []) {
            const foundTool = tools.find(t => t.name === toolCall.name);
            if (!foundTool) continue;

            const toolResult = await foundTool.invoke(toolCall.args);

            // 兼容不同返回格式的字符串化
            const contentStr = typeof toolResult === 'string'
                ? toolResult
                : (toolResult?.text || JSON.stringify(toolResult));

            toolResults.push(new ToolMessage({
                content: contentStr,
                tool_call_id: toolCall.id,
            }));
        }

        return toolResults;
    }
});

// 2. 对结果的处理
const agentStepChain = RunnableSequence.from([
    // step1: 将 LLM 输出挂到 state.response 上
    RunnablePassthrough.assign({
        response: llmChain,
    }),
    // step2: 使用 RunnableBranch 根据是否有 tool_calls 走不同分支
    RunnableBranch.from([
        // 分支1：没有 tool_calls，认为本轮已经完成
        [
            (state) =>
                !state.response?.tool_calls ||
                state.response.tool_calls.length === 0,
            new RunnableLambda({
                func: async (state) => {
                    const { messages, response } = state;
                    const newMessages = [...messages, response];
                    return {
                        ...state,
                        messages: newMessages,
                        done: true,
                        final: response.content,
                    };
                },
            }),
        ],
        // 默认分支：有 tool_calls，调用工具并把 ToolMessage 写回 messages
        RunnableSequence.from([
            new RunnableLambda({
                func: async (state) => {
                    const { messages, response } = state;
                    const newMessages = [...messages, response];

                    console.log(
                        chalk.bgBlue(
                            `🔍 检测到 ${response.tool_calls.length} 个工具调用`
                        )
                    );
                    console.log(
                        chalk.bgBlue(
                            `🔍 工具调用: ${response.tool_calls
                                .map((t) => t.name)
                                .join(', ')}`
                        )
                    );

                    return {
                        ...state,
                        messages: newMessages,
                    };
                },
            }),
            // 调用工具执行器，得到 toolMessages
            RunnablePassthrough.assign({
                toolMessages: toolExecutor,
            }),
            new RunnableLambda({
                func: async (state) => {
                    const { messages, toolMessages } = state;
                    return {
                        ...state,
                        messages: [...messages, ...(toolMessages ?? [])],
                        done: false,
                    };
                },
            }),
        ]),
    ]),
]);

async function runAgentWithTools(query, maxIterations = 30) {
    let state = {
        messages: [new HumanMessage(query)],
        done: false,
        final: null,
        tools,
    };

    for (let i = 0; i < maxIterations; i++) {
        console.log(chalk.bgGreen(`⏳ 正在等待 AI 思考...`));

        // 每一轮都通过一个完整的 Runnable chain（LLM + 工具调用处理）
        state = await agentStepChain.invoke(state);

        if (state.done) {
            console.log(`\n✨ AI 最终回复:\n${state.final}\n`);
            return state.final;
        }
    }

    return state.messages[state.messages.length - 1].content;
}

await runAgentWithTools("杭州滨江江汉路地铁站附近的高口碑饭店，最近的10个饭店，帮我列出饭店名称、评分、距离、价格区间、营业时间、电话、地址，按评分从高到低排序。");

// await mcpClient.close();

// ====================================================================
// 📝 代码问答笔记
// ====================================================================
//
// Q1: tools.find(t => t.name === toolCall.name) 的作用是什么？
// A: 从已注册工具列表中匹配 LLM 想调用的工具。这是防御性检查——
//    LLM 可能幻觉出错误的工具名，find 不到就 continue 跳过，防止崩溃。
//    更好做法是返回错误消息让 LLM 调整策略。
//
// Q2: response 什么情况下有 tool_calls？
// A: LLM 判断需要外部数据或操作时（查天气、搜酒店、打开浏览器）→ 有。
//    LLM 认为能直接回答时（闲聊、知识问答、总结翻译）→ 没有。
//    bindTools 只是告诉 LLM"这些工具可用"，用不用由 LLM 自己决定。

// Q4: RunnablePassthrough.assign({ toolMessages: toolExecutor }) 的作用？
// A: 在 state 上新增 toolMessages 属性（不覆盖原有属性），
//    值由 toolExecutor 惰性求值得出。是"保留原属性 + 追加新属性"的机制。
//
// Q5: agentStepChain 什么时候结束？终止条件是什么？
// A: agentStepChain 本身只是一轮迭代，永远会完整执行完。
//    真正的终止在 runAgentWithTools 的 for 循环中：
//    ① state.done === true（LLM 不再调工具）→ 正常返回
//    ② i >= maxIterations（达到 30 轮上限）→ 强制返回最后一条消息
//
// Q6: 为什么要把 response 加入 messages？
// A: LLM 无状态，每轮需完整上下文。不加的话 LLM 看不到自己上轮的决策。
//    无 tool_calls：保存最终回答供输出。有 tool_calls：形成完整推理链，
//    下一轮 LLM 知道"我调了工具，结果如下"。
//
// Q7: final 和 messages 中的内容是什么关系？
// A: response 本身就是一个 AIMessage 对象。final = response.content，
//    就是 messages 最后一条消息的文本内容。分开存只是方便直接返回。
//
// Q8: 如果 MCP 提供的工具是 ABC，但用户请求需要工具 D，怎么办？
// A: tools.find(t => t.name === "D") → 返回 undefined → continue 跳过。
//    LLM 可能反复尝试直到 maxIterations 耗尽。更好做法是返回错误消息
//    告诉 LLM "D 不存在，可用工具有 A, B, C"，让 LLM 调整策略或组合使用。
// ====================================================================
