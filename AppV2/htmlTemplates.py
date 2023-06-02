css = '''
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    position: relative; /* Added position relative */
}

.chat-message.user {
    background-color: #2b313e;
}

.chat-message.bot {
    background-color: #475063;
}

.chat-message .avatar {
    width: 20%;
}

.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}

.chat-message .message {
    width: 90%;
    padding: 0 1rem;
    color: #fff;
}

.copy-icon {
    position: absolute;
    top: 0.5rem;
    right: 0.5rem;
    font-size: 1.5rem;
    color: #fff;
    cursor: pointer;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://www.sideshow.com/storage/product-images/2171/c-3po_star-wars_square.jpg">
    </div>
    <div class="message">{{MSG}}</div>
    <i class="copy-icon fas fa-copy"></i> <!-- Added copy icon -->
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://resizing.flixster.com/ocuc8yjm8Fu5UK5Ze8lbdp58m9Y=/300x300/v2/https://flxt.tmsimg.com/assets/p11759522_i_h9_aa.jpg">
    </div>
    <div class="message">{{MSG}}</div>
    <i class="copy-icon fas fa-copy"></i> <!-- Added copy icon -->
</div>
'''
