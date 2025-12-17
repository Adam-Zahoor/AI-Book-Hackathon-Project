import React, { useState, useEffect, useRef } from 'react';
import './Chatbot.css';
import { bookContent } from './bookContent';

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { sender: 'bot', text: 'Hello! I\'m your Physical AI and Robotics assistant. Ask me anything about the content in this book.' }
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Function to simulate bot response based on book content
  const getBotResponse = async (userMessage) => {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));

    const lowerCaseMessage = userMessage.toLowerCase();

    // Check for greetings and conversational phrases first
    const greetingPhrases = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening', 'how are you', 'how do you do', 'howdy'];
    const isGreeting = greetingPhrases.some(phrase => lowerCaseMessage.includes(phrase));

    if (isGreeting) {
      return "Hello there! I'm your Physical AI and Robotics assistant. I can help answer questions about the content in the book. What would you like to know about Physical AI, Robotics, or any topic from the book?";
    }

    // Check for thanks responses
    if (lowerCaseMessage.includes('thank') || lowerCaseMessage.includes('thanks') || lowerCaseMessage.includes('appreciate')) {
      return "You're welcome! Feel free to ask more questions about Physical AI and Robotics. I'm here to help with any information from the book.";
    }

    // Find relevant content based on search terms
    // First, extract key terms from the user message
    const keyTerms = userMessage.toLowerCase()
      .replace(/[^\w\s]/g, ' ') // Remove punctuation
      .split(/\s+/) // Split into words
      .filter(word => word.length > 2); // Only consider words with more than 2 characters

    // Search for matches in the content database
    let bestMatch = null;
    let bestScore = 0;

    for (const chapter of bookContent) {
      // Score based on how many key terms appear in the content
      let score = 0;
      const contentLower = chapter.content.toLowerCase();

      for (const term of keyTerms) {
        if (contentLower.includes(term)) {
          score += 1; // Add points for each matching term
        }
      }

      // Additional weight if chapter title matches
      if (chapter.title.toLowerCase().includes(lowerCaseMessage)) {
        score += 5;
      }

      // Add extra weight for exact phrase matches in content
      if (contentLower.includes(lowerCaseMessage)) {
        score += 10;
      }

      // Check for important terms that might be shortened or differently expressed
      if (lowerCaseMessage.includes('physical ai') && contentLower.includes('physical ai')) {
        score += 20;
      }
      if (lowerCaseMessage.includes('machine learning') && contentLower.includes('machine learning')) {
        score += 20;
      }
      if (lowerCaseMessage.includes('robot') && contentLower.includes('robot')) {
        score += 15;
      }

      if (score > bestScore) {
        bestScore = score;
        bestMatch = chapter;
      }
    }

    if (bestMatch && bestScore > 0) {
      // Extract relevant sentences from the best matching chapter
      const sentences = bestMatch.content.split(/\.|\?|!/);
      const relevantSentences = [];

      for (const sentence of sentences) {
        let sentenceScore = 0;
        for (const term of keyTerms) {
          if (sentence.toLowerCase().includes(term)) {
            sentenceScore += 1;
          }
        }

        // Increase score if sentence contains the full query or important terms
        if (sentence.toLowerCase().includes(lowerCaseMessage)) {
          sentenceScore += 10;
        }

        if (sentenceScore > 0) {
          relevantSentences.push({ sentence: sentence.trim(), score: sentenceScore });
        }
      }

      // Sort by score and take top sentences
      relevantSentences.sort((a, b) => b.score - a.score);
      const topSentences = relevantSentences.slice(0, 3).map(item => item.sentence);

      // Formulate response based on the best matching content
      return `Based on "${bestMatch.title}":\n\n${topSentences.join('.\n\n')}.`;
    }

    // If no good match found, provide a general response
    return "I couldn't find specific information about that topic in the book. Our Physical AI and Robotics book covers many topics including AI fundamentals, robotics concepts, hardware platforms, and practical projects. Could you ask about a specific topic like machine learning, robotics fundamentals, hardware, AI integration, or project implementations?";
  };

  const handleSend = async () => {
    if (inputText.trim() === '') return;

    // Add user message
    const newMessage = { sender: 'user', text: inputText };
    setMessages(prev => [...prev, newMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      // Get bot response
      const botResponse = await getBotResponse(inputText);
      // Add bot message
      setMessages(prev => [...prev, { sender: 'bot', text: botResponse }]);
    } catch (error) {
      setMessages(prev => [...prev, { 
        sender: 'bot', 
        text: 'Sorry, I encountered an error processing your request. Please try again.' 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chatbot">
      {!isOpen ? (
        <button className="chatbot-button" onClick={() => setIsOpen(true)}>
          💬
        </button>
      ) : (
        <div className="chatbot-container">
          <div className="chatbot-header">
            <h4>Physical AI Assistant</h4>
            <button className="chatbot-close" onClick={() => setIsOpen(false)}>×</button>
          </div>
          <div className="chatbot-messages">
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.sender}`}>
                <div className="message-text">{msg.text}</div>
              </div>
            ))}
            {isLoading && (
              <div className="message bot">
                <div className="message-text typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <div className="chatbot-input-area">
            <textarea
              className="chatbot-input"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about Physical AI, Robotics, or any topic from the book..."
              rows="2"
            />
            <button className="chatbot-send" onClick={handleSend} disabled={isLoading}>
              Send
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Chatbot;