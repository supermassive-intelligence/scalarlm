import { useCallback, useEffect, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";

import { getApiConfig } from "@/api/config";
import {
  getConversation,
  newConversationId,
  type Conversation,
} from "@/stores/conversations";
import { useConversations } from "@/stores/useConversationStore";

import { ConversationList } from "./ConversationList";
import { ConversationView } from "./ConversationView";

/**
 * Top-level chat route. Sidebar + single-pane layout.
 *
 * Routing:
 *   /chat                    → landing (create a new conversation in memory)
 *   /chat/:conversationId    → specific conversation
 *   ?model=<hash>            → deep link: pre-select a model for the new chat.
 *                              Used by "Open in Chat" from TrainDetail.
 *
 * IndexedDB is the source of truth. Conversations only reach storage once
 * the first user message is sent — this keeps rapid-fire "click New chat"
 * navigations from littering the sidebar.
 */
export function ChatPage() {
  const { conversationId } = useParams<{ conversationId: string }>();
  const [search] = useSearchParams();
  const navigate = useNavigate();
  const conversations = useConversations();

  const [active, setActive] = useState<Conversation | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Resolve the active conversation.
  useEffect(() => {
    let cancelled = false;

    async function resolve() {
      setLoadError(null);
      if (conversationId) {
        const existing = await getConversation(conversationId);
        if (cancelled) return;
        if (!existing) {
          setLoadError(`No conversation ${conversationId}`);
          setActive(null);
          return;
        }
        setActive(existing);
        return;
      }
      // No id in URL — spawn an in-memory draft, not persisted until first send.
      const modelFromQuery = search.get("model");
      const { default_model } = getApiConfig();
      setActive(buildDraft(modelFromQuery ?? default_model));
    }

    resolve();
    return () => {
      cancelled = true;
    };
  }, [conversationId, search]);

  const handleNew = useCallback(() => {
    navigate("/chat");
  }, [navigate]);

  const handleFirstTurn = useCallback(
    async (id: string) => {
      // Persist the draft conversation the first time a user sends a message.
      if (!conversationId) {
        navigate(`/chat/${id}`, { replace: true });
      }
    },
    [conversationId, navigate],
  );

  // Persist draft conversations implicitly when the user edits them via the
  // header title or system-prompt editor BEFORE sending: ConversationView
  // writes through putConversation, which creates the record. Our live list
  // hook then picks it up.
  useEffect(() => {
    if (!conversationId && active && conversations) {
      const stored = conversations.find((c) => c.id === active.id);
      if (stored) navigate(`/chat/${active.id}`, { replace: true });
    }
  }, [active, conversationId, conversations, navigate]);

  return (
    <div className="flex h-full min-h-0">
      <ConversationList onNew={handleNew} />
      <main className="min-h-0 flex-1">
        {loadError ? (
          <div className="flex h-full items-center justify-center px-6 text-sm text-danger">
            {loadError}
          </div>
        ) : active ? (
          <ConversationView
            key={active.id}
            conversation={active}
            onFirstTurn={handleFirstTurn}
          />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-fg-subtle">
            Loading…
          </div>
        )}
      </main>
    </div>
  );
}

function buildDraft(model: string): Conversation {
  const now = Date.now();
  return {
    id: newConversationId(),
    title: "(new chat)",
    model,
    createdAt: now,
    updatedAt: now,
  };
}
