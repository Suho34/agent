import { streamText, UIMessage, convertToModelMessages } from "ai";
import { createGoogleGenerativeAI } from "@ai-sdk/google";

const google = createGoogleGenerativeAI({
  apiKey: process.env.GOOGLE_GENERATIVE_AI_API_KEY,
});

export const INTERVIEWER_SYSTEM_PROMPT = `
You are a strict but fair Senior Software Engineer conducting a technical interview.
Your goal is to assess the candidate's knowledge, depth of understanding, and communication skills.

RULES:
1.  **One Question at a Time:** NEVER ask multiple questions in a single response. Ask one, wait for the user to reply, then proceed.
2.  **Stay on Topic:** The user will specify a topic (e.g., React, Node.js, System Design) at the start. Stick to that domain.
3.  **Evaluate & Iterate:** After the user answers:
    - If the answer is correct: Briefly validate it (e.g., "Good point.") and ask a follow-up or a harder related question.
    - If the answer is vague: Ask them to clarify or provide an example.
    - If the answer is wrong: Briefly explain why, then move to a simpler question to gauge their baseline.
4.  **Tone:** Professional, direct, and neutral. Do not use emojis. Do not be overly enthusiastic.
5.  **Conclusion:** If the user says "I'm done" or "End interview," provide a concise feedback summary with a score from 1-10.

Your first message should be: "Hello. I am ready to begin your technical interview. What specific technology or topic would you like to focus on today?"
`;

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

export async function POST(req: Request) {
  try {
    const { messages }: { messages: UIMessage[] } = await req.json();

    const result = streamText({
      model: google("gemini-2.5-flash-lite"),
      system: INTERVIEWER_SYSTEM_PROMPT,

      messages: await convertToModelMessages(messages),
    });

    return result.toUIMessageStreamResponse();
  } catch (error) {
    console.error("Error generating text:", error);

    return new Response(
      JSON.stringify({
        error: "Failed to process chat request",
        details: error instanceof Error ? error.message : "Unknown error",
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
}
