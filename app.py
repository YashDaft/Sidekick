import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from agentic_chatbot_rag_hitl import chatbot, get_all_threads, ingest_rag_documents
from langgraph.types import Command
import uuid
import os

def extract_text(content):
    """Handle both plain-string content and list-of-block content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = ""
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
        return text
    return ""


def stream_response(user_input, CONFIG):
    for message_chunk, metadata in chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=CONFIG,
        stream_mode="messages"
    ):
        if isinstance(message_chunk, AIMessage):
            text = extract_text(message_chunk.content)
            if text:
                yield text


# generate an unqiue thread id for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):

    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

#create a completely new conversation
def reset_chat():
    # Generate and assign a new thread ID
    st.session_state["thread_id"] = generate_thread_id()

    # Clear message history for the new conversation
    st.session_state['message_history'] = []

    #add the new thread to the conversation
    add_thread(st.session_state['thread_id'])

#loadd a previous conversation from the LangGraph checkpointer
def load_conversation(thread_id):

    #get the saved state for the selected state
    state = chatbot.get_state(
        config={
            'configurable':{
                'thread_id': thread_id
            }
        }
    )

    # return saved messages
    #return an empty list if no messages are available
    return state.values.get("messages", [])


# ---- chat title helpers ----
# instead of showing raw thread ids in the sidebar, derive a short human
# readable title from the first user message of that thread and cache it
# so we don't have to reload the full conversation on every rerun

MAX_TITLE_LENGTH = 40


def make_title_from_text(text):
    text = text.strip().replace("\n", " ")
    if len(text) > MAX_TITLE_LENGTH:
        return text[:MAX_TITLE_LENGTH].rstrip() + "..."
    return text


def get_thread_title(thread_id):
    # return cached title if we already computed one
    if thread_id in st.session_state['chat_titles']:
        return st.session_state['chat_titles'][thread_id]

    # otherwise pull the saved conversation once and derive a title
    # from the first human message found in it
    messages = load_conversation(thread_id)

    for message in messages:
        if isinstance(message, HumanMessage):
            text = extract_text(message.content)
            if text.strip():
                title = make_title_from_text(text)
                st.session_state['chat_titles'][thread_id] = title
                return title

    # no messages yet (brand new / empty thread)
    return "🆕 New Chat"


def set_thread_title_if_missing(thread_id, text):
    """Called right after the first message of a thread is sent, so the
    sidebar label updates immediately instead of waiting for a reload."""
    if thread_id not in st.session_state['chat_titles'] and text.strip():
        st.session_state['chat_titles'][thread_id] = make_title_from_text(text)


# folder where uploaded PDFs are stored before being ingested into the vector store
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_and_ingest_files(uploaded_files):
    """Save each uploaded PDF to disk and ingest it into the RAG vector store.
    Returns a list of filenames that were successfully ingested."""
    ingested_filenames = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

        # save the uploaded file to disk so PyPDFLoader can read it by path
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # feed the saved file straight into the existing ingestion pipeline
        ingest_rag_documents(file_path)

        ingested_filenames.append(uploaded_file.name)

    return ingested_filenames


# ---- human-in-the-loop (HITL) helpers ----
# tools like `purchase_stock` call interrupt(...) inside agentic_chatbot_rag_hitl.py.
# That pauses the graph mid-run instead of finishing it. We detect that paused
# state, show an Approve/Reject UI, and resume the graph with Command(resume=...)
#
# IMPORTANT: this only works if ToolNode is built with handle_tool_errors=False
# in agentic_chatbot_rag_hitl.py. By default ToolNode catches ALL exceptions
# raised inside a tool -- including the special GraphInterrupt that interrupt()
# raises -- and turns it into a normal tool result instead of actually pausing
# the graph. If approvals aren't showing up, that's almost always why.

GENERIC_APPROVAL_NOTICE = "⚠️ This action requires your approval. Use the buttons below to approve or reject."


def build_config(thread_id):
    return {
        "configurable": {
            "thread_id": thread_id,
            "metadata": {"thread_id": thread_id}
        },
        "run_name": "chat_trace",
    }


def get_pending_interrupt(config):
    """Returns the interrupt's message/value if the graph for this thread is
    currently paused waiting on a human decision, otherwise returns None."""
    try:
        state = chatbot.get_state(config)
    except Exception:
        return None

    for task in state.tasks:
        if task.interrupts:
            return task.interrupts[0].value

    return None


def run_and_collect(stream_iterable):
    """Runs a chatbot.stream(...) generator, renders tool-usage status boxes
    and streams AI text into the chat message. Returns the final assistant text."""

    status_holder = {"box": None}

    def _gen():
        for message_chunk, metadata in stream_iterable:

            if isinstance(message_chunk, ToolMessage):
                # ToolMessage stores the tool's name directly on `.name`,
                # NOT on a `.tool` attribute
                tool_name = getattr(message_chunk, "name", None) or "tool"

                if status_holder["box"] is None:
                    status_holder["box"] = st.status(
                        f"🔧 Using `{tool_name}` ...", expanded=True
                    )
                else:
                    status_holder["box"].update(
                        label=f"🔧 Using `{tool_name}` ...", expanded=True
                    )

            if isinstance(message_chunk, AIMessage):
                text = extract_text(message_chunk.content)
                if text:
                    yield text

    result_text = st.write_stream(_gen())

    if status_holder["box"] is not None:
        status_holder["box"].update(
            label="✅ Tool finished", state="complete", expanded=False
        )

    return result_text


def mark_pending_approval(thread_id, interrupt_message):
    """Records the paused thread + question, and drops an explicit notice
    into the chat history so the user sees *why* buttons appeared, instead
    of relying on the LLM to explain it (which it can't, since it never
    got to run again before the pause)."""
    st.session_state['pending_approval'] = {
        'thread_id': thread_id,
        'message': interrupt_message
    }
    st.session_state['message_history'].append({
        'role': 'assistant',
        'content': GENERIC_APPROVAL_NOTICE
    })


#extract the plain text from the

st.title("🤖 Sidekick")


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

#create a list for storing all the conversation thread IDs
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

# cache of thread_id -> display title, so we don't re-derive it every rerun
if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}

# holds {'thread_id': ..., 'message': ...} whenever a thread is paused on
# a human-in-the-loop approval, otherwise None
if "pending_approval" not in st.session_state:
    st.session_state["pending_approval"] = None

add_thread(st.session_state['thread_id'])

st.sidebar.title("💬 My Conversations")

#create a button for starting a new conversation
if st.sidebar.button("➕ New Chat"):

    #reset the current chat and create a new thread
    reset_chat()

    # rerun the streamlit app to update the interface
    st.rerun()

# display all conversation threads in reverse order
# this shows the newest conversation first
for thread_id in st.session_state["chat_threads"][::-1]:

    thread_title = get_thread_title(thread_id)

    #create one sidebar button for every conversation
    if st.sidebar.button(
        f"🗨️ {thread_title}",
        key=thread_id
    ):

        #set the selected thread as the current thread
        st.session_state['thread_id'] = thread_id

        #load the conversation history for this thread
        messages = load_conversation(thread_id)

        # temperory list for converting langchain messages into streamlit's required message format
        temp_messages = []

        # loop though all saved messages

        for message in messages:

            # check wether the message was sent by the user
            if isinstance(message, HumanMessage):
                role = 'user'

            # check wether the message was sent by the ai assistant
            elif isinstance(message, AIMessage):
                role = 'assistant'

            else:
                continue


            # convert the langchain message into a dictionary
            temp_messages.append({
                'role': role,
                'content': extract_text(message.content)
         })

        # replace the current ui history with the selected conversation
        st.session_state['message_history'] = temp_messages

        # re run the messages to display the loaded messages
        st.rerun()


# sync pending_approval with the currently selected thread's actual graph
# state. This runs on every rerun so that switching chats, or refreshing
# the page, correctly re-shows (or hides) the approval prompt.
_current_interrupt = get_pending_interrupt(build_config(st.session_state['thread_id']))

if _current_interrupt:
    if not (
        st.session_state['pending_approval']
        and st.session_state['pending_approval']['thread_id'] == st.session_state['thread_id']
    ):
        # only just discovered this on a fresh page load / thread switch,
        # not something we already flagged during this session
        mark_pending_approval(st.session_state['thread_id'], _current_interrupt)
elif (
    st.session_state['pending_approval']
    and st.session_state['pending_approval']['thread_id'] == st.session_state['thread_id']
):
    st.session_state['pending_approval'] = None


# display all messages from the currently selected conversation

for message in st.session_state['message_history']:

    # create either a user chat bubble or assistant chat buble
    with st.chat_message(message["role"]):

        #display the message content
        st.text(extract_text(message['content']))


# ---- human-in-the-loop approval UI ----
# if the graph for the current thread is paused on interrupt(...), show the
# question along with Approve / Reject buttons instead of letting the user
# type a normal message
pending = st.session_state['pending_approval']
is_awaiting_approval = bool(
    pending and pending['thread_id'] == st.session_state['thread_id']
)

if is_awaiting_approval:

    with st.chat_message('assistant'):

        st.warning(f"🧑 **Human approval required**\n\n{pending['message']}")

        col1, col2 = st.columns(2)
        approve_clicked = col1.button(
            "✅ Approve Purchase",
            type="primary",
            use_container_width=True,
            key=f"approve_{pending['thread_id']}"
        )
        reject_clicked = col2.button(
            "❌ Reject Purchase",
            use_container_width=True,
            key=f"reject_{pending['thread_id']}"
        )

        if approve_clicked or reject_clicked:
            decision = "yes" if approve_clicked else "no"

            CONFIG = build_config(st.session_state['thread_id'])

            # record the human's decision in the visible chat history
            st.session_state['message_history'].append({
                'role': 'user',
                'content': f"({'Approved' if approve_clicked else 'Rejected'}: {decision})"
            })

            with st.spinner("Resuming..."):
                resumed_message = run_and_collect(
                    chatbot.stream(
                        Command(resume=decision),
                        config=CONFIG,
                        stream_mode="messages"
                    )
                )

            if resumed_message:
                st.session_state['message_history'].append({
                    'role': 'assistant',
                    'content': resumed_message
                })

            # clear the current approval, then check whether resuming led
            # straight into another interrupt (e.g. a follow-up approval)
            st.session_state['pending_approval'] = None
            next_interrupt = get_pending_interrupt(CONFIG)
            if next_interrupt:
                mark_pending_approval(st.session_state['thread_id'], next_interrupt)

            st.rerun()


# chat input with a built-in "+" attach icon (accept_file) for uploading PDFs
# directly beside the text box, instead of a separate uploader widget.
# disabled while a human-in-the-loop approval is pending for this thread
user_input = st.chat_input(
    "Type here, or attach a PDF with +",
    accept_file="multiple",
    file_type=["pdf"],
    disabled=is_awaiting_approval,
)

if user_input:

    prompt_text = user_input.text
    uploaded_files = user_input.files

    # handle any PDFs attached via the + icon: save + ingest them into the vector store
    if uploaded_files:
        with st.spinner("📄 Ingesting uploaded document(s)..."):
            ingested_filenames = save_and_ingest_files(uploaded_files)

        confirmation = "✅ Uploaded and indexed: " + ", ".join(ingested_filenames)

        st.session_state['message_history'].append({
            'role': 'assistant',
            'content': confirmation
        })

        with st.chat_message('assistant'):
            st.text(confirmation)

        # if this thread has no title yet, use the first uploaded filename as a fallback title
        set_thread_title_if_missing(
            st.session_state['thread_id'],
            f"📄 {ingested_filenames[0]}"
        )

    # if there's no accompanying text, stop here (upload-only turn)
    if not prompt_text:
        st.stop()

    #save the user's message message in streamlit session state
    st.session_state['message_history'].append(
        {'role':'user' ,
        'content': prompt_text
    })

    # display user's message in the chat interface
    with st.chat_message('user'):
        st.text(prompt_text)

    # if this is the first message of the thread, derive its sidebar title from it
    set_thread_title_if_missing(st.session_state['thread_id'], prompt_text)


    # pass the current thread id to langgraph
    # langgraph uses this id to save and retrieve conversation memory

    CONFIG = build_config(st.session_state['thread_id'])

    #assistant streaming block
    with st.chat_message('assistant'):
        ai_message = run_and_collect(
            chatbot.stream(
                {'messages': [HumanMessage(content=prompt_text)]},
                config=CONFIG,
                stream_mode="messages"
            )
        )

    if ai_message:
        st.session_state['message_history'].append({
            'role': 'assistant',
            'content': ai_message
        })

    # check whether the run stopped because a tool (e.g. purchase_stock)
    # called interrupt(...) and is waiting on a human decision
    interrupt_value = get_pending_interrupt(CONFIG)

    if interrupt_value:
        mark_pending_approval(st.session_state['thread_id'], interrupt_value)
        # rerun so the Approve/Reject UI renders cleanly on its own pass
        st.rerun()
