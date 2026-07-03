"""
Defines the two agents (Retriever, Response Synthesizer) and the two tasks
(retrieval, response) and wires them into a sequential Crew.
"""

from crewai import Agent, Task, Crew, Process, LLM

from rag_tools import DocumentSearchTool, WebSearchTool


def build_llm(model_choice: str) -> LLM:
    """
model_choice: "deepseek-r1" or "qwen2.5"
Both are served locally through Ollama, so make sure you've run:
    ollama pull deepseek-r1
    ollama pull qwen2.5:7b
    ollama serve
"""
    model_map = {
        "deepseek-r1": "ollama/deepseek-r1",
        "qwen2.5": "ollama/qwen2.5:7b",
    }
    return LLM(
        model=model_map[model_choice],
        base_url="http://localhost:11434",
    )


def build_crew(file_path: str, model_choice: str = "llama3.2") -> Crew:
    llm = build_llm(model_choice)

    doc_tool = DocumentSearchTool(file_path=file_path)
    web_tool = WebSearchTool()

    retriever_agent = Agent(
        role="Document Retriever",
        goal=(
            "Find the most relevant information for the user's question. "
            "ALWAYS search the uploaded document first. Only use web_search "
            "if document_search returns NOT_FOUND or the passages don't "
            "actually answer the question."
        ),
        backstory=(
            "You are a meticulous research assistant who never guesses. "
            "You always ground your findings in a real source, either the "
            "uploaded document or a live web result. Note: some documents use "
            "regional title variants (e.g. 'Sorcerer's Stone' vs 'Philosopher's "
            "Stone' are the same book/object, just US vs UK naming) -- treat "
            "such variants as equivalent, not as different things."
        ),
        tools=[doc_tool, web_tool],
        llm=llm,
        verbose=True,
    )

    response_agent = Agent(
        role="Response Synthesizer",
        goal="Turn the retrieved passages into a clear, direct, well-cited answer for the user.",
        backstory=(
            "You are a clear communicator. You never fabricate facts -- "
            "you only use what the Retriever found, and you say so plainly "
            "if the information wasn't available anywhere. Trust the "
            "Retriever's findings even if the exact wording in the source "
            "differs slightly from the question (e.g. synonyms, regional "
            "spelling, or title variants) -- don't invent a contradiction "
            "out of a wording difference."
        ),
        llm=llm,
        verbose=True,
    )

    retrieval_task = Task(
        description=(
            "Answer the following user question by retrieving the most relevant "
            "information: {query}\n\n"
            "You MUST call the document_search tool first, even if you think "
            "you already know the answer -- your own prior knowledge is NOT "
            "a valid source. Only after document_search returns NOT_FOUND or "
            "clearly irrelevant results should you call web_search. "
            "Report exactly what each tool returned, verbatim, plus its source."
        ),
        expected_output="The raw relevant passages/snippets found, with their source (document or web). Never a memorized answer with no tool call.",
        agent=retriever_agent,
    )

    response_task = Task(
        description=(
            "Using ONLY the passages retrieved for: {query}, write a clear, "
            "concise, well-organized final answer. Do NOT add facts from your "
            "own training knowledge that weren't in the retrieved passages -- "
            "if the retrieved passages contradict what you 'remember', trust "
            "the retrieved passages, they are the real source. If nothing "
            "relevant was found anywhere, say so honestly instead of making "
            "something up."
        ),
        expected_output="A final answer in plain, well-formatted markdown, grounded strictly in the retrieved passages.",
        agent=response_agent,
        context=[retrieval_task],
    )

    return Crew(
        agents=[retriever_agent, response_agent],
        tasks=[retrieval_task, response_task],
        process=Process.sequential,
        verbose=True,
    )